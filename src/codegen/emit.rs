use super::*;
use crate::ir::expr::UnOp;
use iced_x86::{Code, Instruction};

/// The width at which a block's control value was computed.
pub(crate) fn control_width(b: &SchedBlock, ctrl: &Operand) -> Width {
    let Operand::Val(want) = ctrl else {
        return Width::W64;
    };
    for i in &b.instrs {
        let (dst, width) = match i {
            Instr::Bin { dst, width, .. }
            | Instr::Un { dst, width, .. }
            | Instr::Select { dst, width, .. }
            | Instr::Load { dst, width, .. } => (dst, *width),
            Instr::Cast { dst, to, .. } => (dst, *to),
            _ => continue,
        };
        if dst == want {
            return width;
        }
    }
    Width::W64
}

/// Emit `test r, r` at `width`, so a narrow value is not read together with the
/// stale bits above it.
pub(crate) fn emit_test_self(asm: &mut CodeAssembler, r: Reg, width: Width) -> Result<()> {
    match width {
        Width::W8 => asm.test(breg(r), breg(r))?,
        Width::W16 => asm.test(wreg(r), wreg(r))?,
        Width::W32 => asm.test(dreg(r), dreg(r))?,
        _ => asm.test(qreg(r), qreg(r))?,
    }
    Ok(())
}

/// Move a value from the architectural register that physically holds it to wherever
/// the allocator homed it.
pub(crate) fn emit_reg_to_home(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    dst: sched::ValId,
    src: Reg,
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    match alloc.loc_at(Item::Val(dst), pos) {
        // Already where the producing instruction left it.
        Some(Loc::Reg(r)) if r == src => {}
        Some(Loc::Reg(r)) => asm.mov(qreg(r), qreg(src))?,
        // Commit spilled definitions to their assigned slot.
        Some(Loc::Spill(i)) => {
            let d = layout.spill_disp(i) as i32;
            asm.mov(qword_ptr(rsp + d), qreg(src))?;
        }
        // No location means no reader; the value is unused.
        None => {}
    }
    Ok(())
}

/// If `pos` is inside a run of `at`-bearing opaques directly following an `Instr::Boxed`, the index of that `Boxed`.
pub(crate) fn boxed_def_group_owner(stream: &[Instr], pos: usize) -> Option<usize> {
    if !matches!(stream.get(pos), Some(Instr::Opaque { at: Some(_), .. })) {
        return None;
    }
    let mut i = pos;
    while i > 0 {
        match &stream[i - 1] {
            Instr::Opaque { at: Some(_), .. } => i -= 1,
            Instr::Boxed { .. } => return Some(i - 1),
            _ => return None,
        }
    }
    None
}

/// Emit the register results of the `Instr::Boxed` at `pos` as one parallel copy.
/// Register moves are ordered to preserve unread sources, with cycles broken by
/// `xchg`; spill stores run before the register permutation.
pub(crate) fn emit_boxed_defs(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    stream: &[Instr],
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    // Spills first: they only read architectural registers, which nothing has touched
    // yet at this point.
    let mut pending: Vec<(Reg, Reg)> = Vec::new(); // (dst, src)
    for (n, instr) in stream[pos + 1..].iter().enumerate() {
        let Instr::Opaque {
            dst, at: Some(src), ..
        } = instr
        else {
            break;
        };
        // The opaque's own position, for `loc_at`. Counted from the run's start rather
        // than from `pending`, which does not advance for a spilled or unused def.
        let at_pos = pos + 1 + n;
        match alloc.loc_at(Item::Val(*dst), at_pos) {
            Some(Loc::Reg(r)) if r == *src => {}
            Some(Loc::Reg(r)) => pending.push((r, *src)),
            Some(Loc::Spill(i)) => {
                let d = layout.spill_disp(i) as i32;
                asm.mov(qword_ptr(rsp + d), qreg(*src))?;
            }
            None => {}
        }
    }

    // Then the register copies, ordered so none destroys a source still to be read.
    // Same algorithm as `emit_call_args`.
    while !pending.is_empty() {
        let ready = pending
            .iter()
            .position(|(d, _)| !pending.iter().any(|(_, s)| s == d));
        match ready {
            Some(i) => {
                let (d, src) = pending.remove(i);
                asm.mov(qreg(d), qreg(src))?;
            }
            None => {
                // Every remaining destination is also a source: a cycle. Swapping the
                // first pair shortens it by one without needing a temporary.
                let (d, src) = pending.remove(0);
                asm.xchg(qreg(d), qreg(src))?;
                for (_, s) in pending.iter_mut() {
                    if *s == d {
                        *s = src;
                    }
                }
                pending.retain(|(d, s)| d != s);
            }
        }
    }
    Ok(())
}

/// Re-emit a boxed instruction: one the VM executed natively because it is outside the subset of AMD64 the interpreter models. These are mostly SSE.
pub(crate) fn emit_boxed(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    site: u64,
    bytes: &[u8],
    mem: Option<&sched::BoxedMemOp>,
    text: &str,
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    if bytes.is_empty() {
        bail!("boxed instruction at {site:#x} ({text}) has no recorded bytes");
    }

    let mut dec = iced_x86::Decoder::with_ip(64, bytes, site, iced_x86::DecoderOptions::NONE);
    let mut inst = dec.decode();
    if inst.is_invalid() {
        bail!("boxed instruction at {site:#x} ({text}) did not decode");
    }

    let Some(m) = mem else {
        // No memory operand: the encoding is self-contained. `cli`, `xorps xmm,xmm` and friends name only registers the allocator does not manage.
        if !inst.is_string_instruction()
            && !seg_relative_mem(&inst)
            && (0..inst.op_count()).any(|i| is_memory_op_kind(inst.op_kind(i)))
        {
            bail!(
                "boxed instruction at {site:#x} ({text}) has a memory operand with no recorded address"
            );
        }
        emit_encoded(asm, &inst, site, text)?;
        return Ok(());
    };

    // A scratch register to hold the address. It must not be one the instruction
    // itself names, or rewriting the base would clobber an operand.
    let mut avoid: Vec<Reg> = Vec::new();
    for i in 0..inst.op_count() {
        if inst.op_kind(i) == iced_x86::OpKind::Register {
            if let Some(r) = reg_of_iced(inst.op_register(i)) {
                avoid.push(r);
            }
        }
    }
    for r in [inst.memory_base(), inst.memory_index()] {
        if let Some(r) = reg_of_iced(r) {
            avoid.push(r);
        }
    }

    let scratch = alloc.free_volatile_at(pos, &avoid);
    let base = scratch.unwrap_or(Reg::Rax);
    let borrowed = scratch.is_none();

    // Borrowing rax means parking it in the frame first. `push` is not an option:
    // it moves RSP and every RSP-relative operand in this computation, including
    // the address about to be materialized, would shift under it.
    if borrowed {
        let d = layout.scratch_disp();
        asm.mov(qword_ptr(rsp + d), qreg(Reg::Rax))?;
    }

    materialize_address(asm, alloc, &m.addr, base, pos, layout)?;

    // Point the operand at [base] with nothing else contributing.
    inst.set_memory_base(iced_of_reg(base));
    inst.set_memory_index(iced_x86::Register::None);
    inst.set_memory_index_scale(1);
    inst.set_memory_displacement64(0);
    inst.set_memory_displ_size(0);
    inst.set_is_broadcast(false);
    emit_encoded(asm, &inst, site, text)?;

    if borrowed {
        let d = layout.scratch_disp();
        asm.mov(qreg(Reg::Rax), qword_ptr(rsp + d))?;
    }
    Ok(())
}

/// Whether an operand kind reads or writes memory.
///
/// Wider than `== OpKind::Memory`: the `MemorySeg*` and `MemoryES*` kinds are the
/// implicit string-operand addressings, where the base register is not encoded and
/// `set_memory_base` cannot redirect it.
pub(crate) fn is_memory_op_kind(k: iced_x86::OpKind) -> bool {
    use iced_x86::OpKind as K;
    matches!(
        k,
        K::Memory
            | K::MemorySegSI
            | K::MemorySegESI
            | K::MemorySegRSI
            | K::MemorySegDI
            | K::MemorySegEDI
            | K::MemorySegRDI
            | K::MemoryESDI
            | K::MemoryESEDI
            | K::MemoryESRDI
    )
}

/// Whether the instruction uses an FS/GS-relative memory operand.
pub(crate) fn seg_relative_mem(inst: &iced_x86::Instruction) -> bool {
    use iced_x86::Register as R;
    matches!(inst.segment_prefix(), R::FS | R::GS)
        && (0..inst.op_count()).any(|i| is_memory_op_kind(inst.op_kind(i)))
}
/// Encode one instruction and emit its bytes directly.
pub(crate) fn emit_encoded(
    asm: &mut CodeAssembler,
    inst: &iced_x86::Instruction,
    site: u64,
    text: &str,
) -> Result<()> {
    let mut inst = *inst;
    inst.set_ip(0);
    asm.add_instruction(inst).map_err(|e| {
        anyhow::anyhow!("boxed instruction at {site:#x} ({text}) failed to encode: {e}")
    })?;
    Ok(())
}

/// Compute a boxed operand's address into `dst`.
pub(crate) fn materialize_address(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    addr: &Operand,
    dst: Reg,
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    match addr {
        // An image address is materialized RIP-relative, for the reason `rip_lea` documents, no `.reloc` entry is ever added, so an absolute immediate names the wrong byte once the loader rebases the driver.
        Operand::Imm(a) if layout.img.rip_ok(*a) => {
            rip_lea(asm, dst, *a)?;
        }
        Operand::Imm(a) => {
            asm.mov(qreg(dst), *a)?;
        }
        Operand::Frame(off) => {
            let d = layout.frame_disp(*off);
            asm.lea(qreg(dst), qword_ptr(rsp + d))?;
        }
        Operand::OutArg(n) => {
            asm.lea(qreg(dst), qword_ptr(rsp + *n as i64))?;
        }
        Operand::Val(_) | Operand::Param(_) | Operand::Entry(_) => {
            let src = ensure_reg(asm, alloc, addr, pos, Width::W64, dst, layout)?;
            if src != dst {
                asm.mov(qreg(dst), qreg(src))?;
            }
        }
    }
    Ok(())
}

/// The `Reg` an iced register maps to, for the general-purpose 64-bit set only.
///
/// xmm and the rest return `None`: they are not allocator-managed, so they neither
/// need avoiding nor can be materialized into.
pub(crate) fn reg_of_iced(r: iced_x86::Register) -> Option<Reg> {
    Some(match r.full_register() {
        iced_x86::Register::RAX => Reg::Rax,
        iced_x86::Register::RCX => Reg::Rcx,
        iced_x86::Register::RDX => Reg::Rdx,
        iced_x86::Register::RBX => Reg::Rbx,
        iced_x86::Register::RSP => Reg::Rsp,
        iced_x86::Register::RBP => Reg::Rbp,
        iced_x86::Register::RSI => Reg::Rsi,
        iced_x86::Register::RDI => Reg::Rdi,
        iced_x86::Register::R8 => Reg::R8,
        iced_x86::Register::R9 => Reg::R9,
        iced_x86::Register::R10 => Reg::R10,
        iced_x86::Register::R11 => Reg::R11,
        iced_x86::Register::R12 => Reg::R12,
        iced_x86::Register::R13 => Reg::R13,
        iced_x86::Register::R14 => Reg::R14,
        iced_x86::Register::R15 => Reg::R15,
        _ => return None,
    })
}

/// The iced register for a `Reg`, 64-bit.
pub(crate) fn iced_of_reg(r: Reg) -> iced_x86::Register {
    match r {
        Reg::Rax => iced_x86::Register::RAX,
        Reg::Rcx => iced_x86::Register::RCX,
        Reg::Rdx => iced_x86::Register::RDX,
        Reg::Rbx => iced_x86::Register::RBX,
        Reg::Rsp => iced_x86::Register::RSP,
        Reg::Rbp => iced_x86::Register::RBP,
        Reg::Rsi => iced_x86::Register::RSI,
        Reg::Rdi => iced_x86::Register::RDI,
        Reg::R8 => iced_x86::Register::R8,
        Reg::R9 => iced_x86::Register::R9,
        Reg::R10 => iced_x86::Register::R10,
        Reg::R11 => iced_x86::Register::R11,
        Reg::R12 => iced_x86::Register::R12,
        Reg::R13 => iced_x86::Register::R13,
        Reg::R14 => iced_x86::Register::R14,
        Reg::R15 => iced_x86::Register::R15,
    }
}

/// Commit a value the allocator placed in a spill slot.
///
/// The computation uses a volatile scratch register and preserves it in the frame
/// when no register is free. RSP remains unchanged so frame-relative operands stay
/// valid.
pub(crate) fn with_spilled_dst(
    asm: &mut CodeAssembler,
    layout: &FrameLayout,
    slot: u32,
    width: Width,
    alloc: &Alloc,
    pos: usize,
    body: impl FnOnce(&mut CodeAssembler, Reg) -> Result<()>,
) -> Result<()> {
    let dest = layout.spill_disp(slot) as i32;

    let (scratch, must_preserve) = match alloc.free_volatile_at(pos, &[]) {
        Some(r) => (r, false),
        None => (Reg::Rax, true),
    };
    let keep = layout.scratch_disp() as i32;

    if must_preserve {
        asm.mov(qword_ptr(rsp + keep), qreg(scratch))?;
    }
    body(asm, scratch)?;
    match width {
        Width::W64 => asm.mov(qword_ptr(rsp + dest), qreg(scratch))?,
        Width::W32 => asm.mov(dword_ptr(rsp + dest), dreg(scratch))?,
        Width::W16 => asm.mov(word_ptr(rsp + dest), wreg(scratch))?,
        Width::W8 => asm.mov(byte_ptr(rsp + dest), breg(scratch))?,
    }
    if must_preserve {
        asm.mov(qreg(scratch), qword_ptr(rsp + keep))?;
    }
    Ok(())
}

/// Ensure `op` is in a register, loading it from a spill slot into `scratch`
/// if needed. Returns the register actually holding the value.
pub(crate) fn ensure_reg(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    op: &Operand,
    pos: usize,
    width: Width,
    scratch: Reg,
    layout: &FrameLayout,
) -> Result<Reg> {
    match resolve(alloc, op, pos) {
        Some(Loc::Reg(r)) => Ok(r),
        Some(Loc::Spill(i)) => {
            let d = layout.spill_disp(i) as i32;
            match width {
                Width::W64 => asm.mov(qreg(scratch), qword_ptr(rsp + d))?,
                Width::W16 | Width::W32 => asm.mov(dreg(scratch), dword_ptr(rsp + d))?,
                Width::W8 => asm.mov(breg(scratch), byte_ptr(rsp + d))?,
            }
            Ok(scratch)
        }
        None => {
            match op {
                Operand::Imm(c) => emit_mov_imm(asm, scratch, *c, width, layout)?,
                Operand::Frame(o) => {
                    asm.lea(qreg(scratch), qword_ptr(rsp + layout.frame_disp(*o) as i32))?;
                }
                // Anchored to the RSP the call will see, so the displacement is the
                // offset, with none of `frame_disp`'s entry-RSP correction.
                Operand::OutArg(n) => {
                    asm.lea(qreg(scratch), qword_ptr(rsp + *n as i64))?;
                }
                Operand::Val(_) | Operand::Param(_) | Operand::Entry(_) => {
                    bail!("no location for {op:?} at position {pos}")
                }
            }
            Ok(scratch)
        }
    }
}

/// Emit one scheduled instruction into `asm`. `scratch_a`/`scratch_b` are caller-provided temporaries that this function may clobber to load spilled values. Returns `true` if a comparison was emitted and should be fused into the terminator Jcc (i.e. the instruction is the `control` definer).
pub(crate) fn emit_instr(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    instr: &Instr,
    pos: usize,
    layout: &FrameLayout,
    img: ImageRange,
    // The instruction stream this one belongs to, for the cases that must look at
    // their neighbours. `None` in unit tests that emit a single instruction.
    stream: Option<&[Instr]>,
    redirect: &CallRedirects,
) -> Result<()> {
    match instr {
        Instr::Opaque {
            dst,
            tag,
            width,
            at,
        } => {
            // Values with an architectural home already exist in `at`; move them to
            // the allocator-selected location. Boxed result groups are handled by the
            // preceding `Boxed` instruction as a parallel copy.
            let Some(src) = at else { return Ok(()) };
            // Boxed result groups are emitted together to preserve parallel-copy
            // semantics.
            if let Some(stream) = stream
                && boxed_def_group_owner(stream, pos).is_some()
            {
                return Ok(());
            }
            emit_reg_to_home(asm, alloc, *dst, *src, pos, layout)?;
            let _ = (tag, width);
        }
        Instr::Boxed {
            site,
            bytes,
            mem,
            text,
            uses,
        } => {
            // The implicit register reads first: `cpuid`'s leaf in EAX, `rdmsr`'s MSR index in ECX, `rep stosb`'s RDI/RCX/AL. The re-emitted encoding names the VM's registers, so without this the instruction runs against whatever the allocator left there.
            emit_call_args(asm, alloc, uses, pos, layout)?;
            emit_boxed(asm, alloc, *site, bytes, mem.as_ref(), text, pos, layout)?;
            // The register results this instruction just produced, as one parallel copy.
            if let Some(stream) = stream {
                emit_boxed_defs(asm, alloc, stream, pos, layout)?;
            }
        }

        Instr::Cast {
            dst,
            kind,
            a,
            from,
            to,
        } => {
            let dst_loc = alloc.loc_at(Item::Val(*dst), pos);
            let src_loc = resolve(alloc, a, pos);
            match dst_loc {
                Some(Loc::Reg(_)) | None => {
                    emit_cast(asm, *kind, dst_loc, src_loc, a, *from, *to, layout)?;
                }
                Some(Loc::Spill(i)) => {
                    with_spilled_dst(asm, layout, i, *to, alloc, pos, |asm, scratch| {
                        emit_cast(
                            asm,
                            *kind,
                            Some(Loc::Reg(scratch)),
                            src_loc,
                            a,
                            *from,
                            *to,
                            layout,
                        )
                    })?
                }
            }
        }

        Instr::Un { dst, op, a, width } => match alloc.loc_at(Item::Val(*dst), pos) {
            Some(Loc::Reg(r)) => emit_un_into(asm, alloc, r, *op, a, *width, pos, layout)?,
            Some(Loc::Spill(i)) => {
                with_spilled_dst(asm, layout, i, *width, alloc, pos, |asm, scratch| {
                    emit_un_into(asm, alloc, scratch, *op, a, *width, pos, layout)
                })?
            }
            None => {}
        },
        Instr::Bin {
            dst,
            op,
            a,
            b,
            width,
            operand_width,
        } => {
            let w = match op {
                crate::ir::expr::BinOp::Eq
                | crate::ir::expr::BinOp::Ult
                | crate::ir::expr::BinOp::Slt => *operand_width,
                _ => *width,
            };
            emit_bin(asm, alloc, *dst, *op, a, b, w, *width, pos, layout)?;
        }

        Instr::Load {
            dst,
            addr,
            disp,
            width,
        } => match alloc.loc_at(Item::Val(*dst), pos) {
            Some(Loc::Reg(r)) => emit_load(asm, alloc, r, addr, *disp, *width, pos, layout, img)?,
            Some(Loc::Spill(i)) => {
                with_spilled_dst(asm, layout, i, *width, alloc, pos, |asm, scratch| {
                    emit_load(asm, alloc, scratch, addr, *disp, *width, pos, layout, img)
                })?
            }
            None => {}
        },

        Instr::Store {
            addr,
            value,
            disp,
            width,
        } => {
            emit_store(asm, alloc, addr, value, *disp, *width, pos, layout, img)?;
        }

        Instr::Call { target, args } => {
            emit_call_args(asm, alloc, args, pos, layout)?;
            emit_call(asm, alloc, target, pos, img, layout, redirect)?;
        }
        Instr::Select {
            dst,
            cond,
            a,
            b,
            width,
        } => match alloc.loc_at(Item::Val(*dst), pos) {
            Some(Loc::Reg(r)) => emit_select_into(asm, alloc, r, cond, a, b, *width, pos, layout)?,
            Some(Loc::Spill(i)) => {
                with_spilled_dst(asm, layout, i, *width, alloc, pos, |asm, scratch| {
                    emit_select_into(asm, alloc, scratch, cond, a, b, *width, pos, layout)
                })?
            }
            None => {}
        },
    }
    Ok(())
}

/// Compute a select into `dst_r`, using a scratch register when the destination is
/// spilled.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_select_into(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    dst_r: Reg,
    cond: &Operand,
    a: &Operand,
    b: &Operand,
    width: Width,
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    // Choose a scratch for `b` that does not alias `dst_r` or RSP.
    let b_scratch =
        alloc
            .free_volatile_at(pos, &[dst_r, Reg::Rsp])
            .unwrap_or(if dst_r != Reg::Rax {
                Reg::Rax
            } else {
                Reg::Rcx
            });
    // Choose a scratch for `cond` from what remains.
    let c_scratch = alloc
        .free_volatile_at(pos, &[dst_r, b_scratch, Reg::Rsp])
        .unwrap_or(if dst_r != Reg::Rcx && b_scratch != Reg::Rcx {
            Reg::Rcx
        } else {
            Reg::Rdx
        });
    let a_r = ensure_reg(asm, alloc, a, pos, width, dst_r, layout)?;
    if a_r != dst_r {
        match width {
            Width::W64 => asm.mov(qreg(dst_r), qreg(a_r))?,
            _ => asm.mov(dreg(dst_r), dreg(a_r))?,
        }
    }
    let b_r = ensure_reg(asm, alloc, b, pos, width, b_scratch, layout)?;
    let c_r = ensure_reg(asm, alloc, cond, pos, Width::W8, c_scratch, layout)?;
    // `dst` already holds `a`, so only the false case needs a move. [`Arena::select`] defines the condition as `cond != 0 -> a`, hence CMOVZ: take `b` exactly when the condition tested zero.
    asm.test(breg(c_r), breg(c_r))?;
    match width {
        Width::W64 => asm.cmovz(qreg(dst_r), qreg(b_r))?,
        _ => asm.cmovz(dreg(dst_r), dreg(b_r))?,
    }
    Ok(())
}

pub(crate) fn emit_cast(
    asm: &mut CodeAssembler,
    kind: CastKind,
    dst_loc: Option<Loc>,
    src_loc: Option<Loc>,
    src_op: &Operand,
    from: Width,
    to: Width,
    layout: &FrameLayout,
) -> Result<()> {
    let Some(Loc::Reg(dst_r)) = dst_loc else {
        return Ok(());
    };
    // Immediate / Frame source: materialise it, then treat as already-extended.
    if src_loc.is_none() {
        emit_mov_operand_imm_or_frame(asm, dst_r, src_op, to, layout)?;
        return Ok(());
    }
    let src_r = match src_loc {
        Some(Loc::Reg(r)) => r,
        Some(Loc::Spill(i)) => {
            // Load into dst_r first at the *source* width, then extend in place.
            let d = layout.spill_disp(i) as i32;
            match from {
                Width::W64 => asm.mov(qreg(dst_r), qword_ptr(rsp + d))?,
                Width::W16 | Width::W32 => asm.mov(dreg(dst_r), dword_ptr(rsp + d))?,
                Width::W8 => asm.mov(breg(dst_r), byte_ptr(rsp + d))?,
            }
            dst_r
        }
        None => unreachable!(),
    };
    // Widening casts use the opcode table. W32-to-W64 zero extension writes the
    // 32-bit destination, which clears the upper half.
    if let Some((code, dst_w)) = cast_code(kind, from, to) {
        asm.add_instruction(iced_x86::Instruction::with2(
            code,
            ireg(dst_r, dst_w),
            ireg(src_r, from),
        )?)?;
        return Ok(());
    }

    // Cases not handled by the table:
    //
    // - Same-width Zext/Sext: the original value is already the right width, so a
    //   plain full-width copy (or nothing if already in place).
    // - Trunc: a copy at the *destination* width. W16 rides the 32-bit move form.
    // - Narrowing Zext/Sext (e.g. W64 → W32): genuinely invalid; the scheduler emits
    //   those as Trunc.
    //
    // The `Mov_rm*_r*` forms match what the fluent `asm.mov(qreg(x), qreg(y))` emits
    // (opcode 0x89), as confirmed by the castprobe example.
    match kind {
        CastKind::Trunc => {
            if src_r != dst_r {
                use iced_x86::Code as C;
                let (code, rw) = match to {
                    Width::W64 => (C::Mov_rm64_r64, Width::W64),
                    // W16 values travel in 32-bit registers; every consumer reads
                    // through wreg, so the upper bits are never observed.
                    Width::W32 | Width::W16 => (C::Mov_rm32_r32, Width::W32),
                    Width::W8 => (C::Mov_rm8_r8, Width::W8),
                };
                asm.add_instruction(iced_x86::Instruction::with2(
                    code,
                    ireg(dst_r, rw),
                    ireg(src_r, rw),
                )?)?;
            }
        }
        _ if from == to => {
            if src_r != dst_r {
                asm.add_instruction(iced_x86::Instruction::with2(
                    iced_x86::Code::Mov_rm64_r64,
                    ireg(dst_r, Width::W64),
                    ireg(src_r, Width::W64),
                )?)?;
            }
        }
        _ => bail!("{kind:?} from {from:?} to {to:?} is a narrowing cast"),
    }
    Ok(())
}

/// The `Code` and destination register width for a widening cast. Returns `None` for a same-width or narrowing cast: the caller handles the former as a plain copy and rejects the latter, and neither needs a table entry.
pub(crate) fn cast_code(kind: CastKind, from: Width, to: Width) -> Option<(iced_x86::Code, Width)> {
    use iced_x86::Code as C;
    Some(match (kind, from, to) {
        (CastKind::Zext, Width::W8, Width::W64) => (C::Movzx_r64_rm8, Width::W64),
        (CastKind::Zext, Width::W8, Width::W32) => (C::Movzx_r32_rm8, Width::W32),
        (CastKind::Zext, Width::W8, Width::W16) => (C::Movzx_r16_rm8, Width::W16),
        (CastKind::Zext, Width::W16, Width::W64) => (C::Movzx_r64_rm16, Width::W64),
        (CastKind::Zext, Width::W16, Width::W32) => (C::Movzx_r32_rm16, Width::W32),
        // Both operands 32-bit; the zeroing of the upper half is the extension.
        (CastKind::Zext, Width::W32, Width::W64) => (C::Mov_rm32_r32, Width::W32),
        (CastKind::Sext, Width::W8, Width::W64) => (C::Movsx_r64_rm8, Width::W64),
        (CastKind::Sext, Width::W8, Width::W32) => (C::Movsx_r32_rm8, Width::W32),
        (CastKind::Sext, Width::W8, Width::W16) => (C::Movsx_r16_rm8, Width::W16),
        (CastKind::Sext, Width::W16, Width::W64) => (C::Movsx_r64_rm16, Width::W64),
        (CastKind::Sext, Width::W16, Width::W32) => (C::Movsx_r32_rm16, Width::W32),
        (CastKind::Sext, Width::W32, Width::W64) => (C::Movsxd_r64_rm32, Width::W64),
        _ => return None,
    })
}

pub(crate) fn fits_imm32(c: u64) -> bool {
    c as i64 == c as i32 as i64 // 
}

/// Materialise `c` into `dst` at `width`.
pub(crate) fn emit_mov_imm(
    asm: &mut CodeAssembler,
    dst: Reg,
    c: u64,
    width: Width,
    layout: &FrameLayout,
) -> Result<()> {
    match width {
        Width::W64 => {
            if c == 0 {
                asm.xor(dreg(dst), dreg(dst))?;
            } else if layout.img.rip_ok(c) {
                // An image address: keep it relocation-independent.
                rip_lea(asm, dst, c)?;
            } else if c <= u32::MAX as u64 {
                asm.mov(dreg(dst), c as u32)?;
            } else {
                asm.mov(qreg(dst), c as i64)?;
            }
        }
        Width::W16 | Width::W32 => {
            if c as u32 == 0 {
                asm.xor(dreg(dst), dreg(dst))?;
            } else {
                asm.mov(dreg(dst), c as u32)?;
            }
        }
        Width::W8 => asm.mov(breg(dst), c as u8 as u32)?,
    }
    Ok(())
}

pub(crate) fn emit_mov_operand_imm_or_frame(
    asm: &mut CodeAssembler,
    dst: Reg,
    op: &Operand,
    width: Width,
    layout: &FrameLayout,
) -> Result<()> {
    match op {
        Operand::Imm(c) => emit_mov_imm(asm, dst, *c, width, layout)?,
        Operand::Frame(o) => {
            asm.lea(qreg(dst), qword_ptr(rsp + layout.frame_disp(*o) as i32))?;
        }
        _ => {}
    }
    Ok(())
}

/// Apply a fluent `CodeAssembler` shift/rotate method at a runtime width. The five shift opcodes each had four near-identical immediate arms differing only in the register view.
macro_rules! shift_by_imm {
    ($asm:expr, $method:ident, $dst:expr, $count:expr, $width:expr) => {
        match $width {
            Width::W64 => $asm.$method(qreg($dst), $count)?,
            Width::W32 => $asm.$method(dreg($dst), $count)?,
            Width::W16 => $asm.$method(wreg($dst), $count)?,
            Width::W8 => $asm.$method(breg($dst), $count)?,
        }
    };
}

/// Emit the `setcc` that materializes a comparison's boolean into `dst_r`.
pub(crate) fn emit_setcc(
    asm: &mut CodeAssembler,
    op: crate::ir::expr::BinOp,
    dst_r: Reg,
) -> Result<()> {
    use crate::ir::expr::BinOp::*;
    match op {
        Eq => asm.sete(breg(dst_r))?,
        Ult => asm.setb(breg(dst_r))?,
        // Signed, so SF/OF rather than CF.
        Slt => asm.setl(breg(dst_r))?,
        other => bail!("{other:?} is not a comparison"),
    }
    Ok(())
}

/// Emit `cmp <dst_r at width>, imm` then the `setcc` for `op`. The caller guards the W64 case on `fits_imm32`; an immediate that does not fit falls through to the register path, which materializes it first.
pub(crate) fn emit_cmp_imm_setcc(
    asm: &mut CodeAssembler,
    op: crate::ir::expr::BinOp,
    dst_r: Reg,
    c: u64,
    width: Width,
) -> Result<()> {
    match width {
        Width::W64 => asm.cmp(qreg(dst_r), c as i32)?,
        Width::W32 => asm.cmp(dreg(dst_r), c as i32 as u32)?,
        Width::W16 => asm.cmp(wreg(dst_r), c as u16 as i32)?,
        Width::W8 => asm.cmp(breg(dst_r), c as u8 as u32)?,
    }
    emit_setcc(asm, op, dst_r)
}

/// Iced `Code` values for the two-register ALU forms, one per width.
///
/// `b` is `None` when the instruction has no byte form. Byte multiplication uses the
/// dword form because the low byte of the product depends only on the low bytes of
/// its operands.
pub(crate) struct BinRegCodes {
    q: iced_x86::Code,
    d: iced_x86::Code,
    w: iced_x86::Code,
    b: Option<iced_x86::Code>,
}

impl BinRegCodes {
    fn for_width(&self, w: Width) -> Option<iced_x86::Code> {
        Some(match w {
            Width::W64 => self.q,
            Width::W32 => self.d,
            Width::W16 => self.w,
            Width::W8 => self.b?,
        })
    }
}

/// The `Code` table for `op`, or `None` for the opcodes that cannot take the "move `a` into the destination, then operate" shape.
pub(crate) fn bin_reg_codes(op: crate::ir::expr::BinOp) -> Option<BinRegCodes> {
    use crate::ir::expr::BinOp::*;
    use iced_x86::Code as C;
    Some(match op {
        Add => BinRegCodes {
            q: C::Add_rm64_r64,
            d: C::Add_rm32_r32,
            w: C::Add_rm16_r16,
            b: Some(C::Add_rm8_r8),
        },
        Sub => BinRegCodes {
            q: C::Sub_rm64_r64,
            d: C::Sub_rm32_r32,
            w: C::Sub_rm16_r16,
            b: Some(C::Sub_rm8_r8),
        },
        And => BinRegCodes {
            q: C::And_rm64_r64,
            d: C::And_rm32_r32,
            w: C::And_rm16_r16,
            b: Some(C::And_rm8_r8),
        },
        Or => BinRegCodes {
            q: C::Or_rm64_r64,
            d: C::Or_rm32_r32,
            w: C::Or_rm16_r16,
            b: Some(C::Or_rm8_r8),
        },
        Xor => BinRegCodes {
            q: C::Xor_rm64_r64,
            d: C::Xor_rm32_r32,
            w: C::Xor_rm16_r16,
            b: Some(C::Xor_rm8_r8),
        },
        // `imul` has only the `r, r/m` two-operand form; there is no `r/m, r` variant,
        // so this one keeps that shape and its bytes are unchanged either way.
        Mul => BinRegCodes {
            q: C::Imul_r64_rm64,
            d: C::Imul_r32_rm32,
            w: C::Imul_r16_rm16,
            b: None,
        },
        Shl | Shr | Sar | Rol | Ror => return None,
        MulHiU | MulHiS | UDiv | URem | SDiv | SRem => return None,
        Eq | Ult | Slt => return None,
    })
}

/// Emit: `dst = a op b` at `width`. Strategy: copy `a` into `dst` if not already there, then apply the operation against `b`. For immediates and Frame refs `b` is materialised into a scratch register (rax if free, else whatever alloc left there).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_un_into(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    dst_r: Reg,
    op: crate::ir::expr::UnOp,
    a: &Operand,
    width: Width,
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    let src_r = ensure_reg(asm, alloc, a, pos, width, dst_r, layout)?;
    if src_r != dst_r {
        match width {
            Width::W64 => asm.mov(qreg(dst_r), qreg(src_r))?,
            Width::W16 | Width::W32 => asm.mov(dreg(dst_r), dreg(src_r))?,
            Width::W8 => asm.mov(breg(dst_r), breg(src_r))?,
        }
    }
    match (op, width) {
        (UnOp::Not, Width::W64) => asm.not(qreg(dst_r))?,
        (UnOp::Not, Width::W16 | Width::W32) => asm.not(dreg(dst_r))?,
        (UnOp::Not, Width::W8) => asm.not(breg(dst_r))?,
        (UnOp::Neg, Width::W64) => asm.neg(qreg(dst_r))?,
        (UnOp::Neg, Width::W16 | Width::W32) => asm.neg(dreg(dst_r))?,
        (UnOp::Neg, Width::W8) => asm.neg(breg(dst_r))?,
        // x86's PF is the even parity of the low byte, which is exactly what `test r8, r8` computes, so `setp` reads it straight back out.
        (UnOp::ParityByte, _) => {
            asm.test(breg(dst_r), breg(dst_r))?;
            asm.setp(breg(dst_r))?;
        }
        (UnOp::Bswap, Width::W64) => asm.bswap(qreg(dst_r))?,
        (UnOp::Bswap, Width::W32) => asm.bswap(dreg(dst_r))?,
        // `bswap` has no 8- or 16-bit form. A byte swap of one byte is the
        // identity; W16 would need `rol r16, 8` and has never been seen, so
        // it fails loudly rather than silently emitting the wrong thing.
        (UnOp::Bswap, Width::W8) => {}
        (UnOp::Bswap, Width::W16) => bail!("16-bit Bswap is not implemented"),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_bin(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    dst: crate::ir::sched::ValId,
    op: crate::ir::expr::BinOp,
    a: &Operand,
    b: &Operand,
    width: Width,
    result_width: Width,
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    match alloc.loc_at(Item::Val(dst), pos) {
        Some(Loc::Reg(r)) => emit_bin_into(asm, alloc, r, op, a, b, width, pos, layout),
        Some(Loc::Spill(i)) => {
            with_spilled_dst(asm, layout, i, result_width, alloc, pos, |asm, scratch| {
                emit_bin_into(asm, alloc, scratch, op, a, b, width, pos, layout)
            })
        }
        // No home at all: the allocator placed nothing, so there is nothing to write.
        None => Ok(()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_bin_into(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    dst_r: Reg,
    op: crate::ir::expr::BinOp,
    a: &Operand,
    b: &Operand,
    width: Width,
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    use crate::ir::expr::BinOp::*;

    // The opcodes that x86 only offers on the fixed RDX:RAX pair. They cannot take the
    // "move `a` into the destination, then operate" shape the rest of this function
    // uses, so they are dispatched before any of it runs.
    if matches!(op, MulHiU | MulHiS | UDiv | URem | SDiv | SRem) {
        return emit_rdx_rax_op(asm, alloc, dst_r, op, a, b, width, pos, layout);
    }

    if width == Width::W64 && matches!(op, Add | And | Or | Xor | Mul) {
        let needs_materializing = match b {
            Operand::Imm(c) => !fits_imm32(*c),
            // `Add` is the only one of these that is meaningful on an address, but the
            // rewrite is valid for any of them.
            Operand::Frame(_) => true,
            _ => false,
        };
        if needs_materializing {
            if let Some(Loc::Reg(a_r)) = resolve(alloc, a, pos) {
                if a_r != dst_r {
                    emit_mov_operand_imm_or_frame(asm, dst_r, b, width, layout)?;
                    match op {
                        Add => asm.add(qreg(dst_r), qreg(a_r))?,
                        And => asm.and(qreg(dst_r), qreg(a_r))?,
                        Or => asm.or(qreg(dst_r), qreg(a_r))?,
                        Xor => asm.xor(qreg(dst_r), qreg(a_r))?,
                        Mul => asm.imul_2(qreg(dst_r), qreg(a_r))?,
                        _ => unreachable!("guarded by the `matches!` above"),
                    }
                    return Ok(());
                }
            }
        }
    }

    // `b` may already live in `dst_r`, and the next step overwrites `dst_r` with `a`.
    let a_loc = resolve(alloc, a, pos);
    let mut b_relocated: Option<Reg> = None;
    let mut restore_b_holder: Option<Reg> = None;
    if resolve(alloc, b, pos) == Some(Loc::Reg(dst_r)) && a_loc != Some(Loc::Reg(dst_r)) {
        // Anything holding `a` is excluded: `a` is still live and is read below.
        let avoid: Vec<Reg> = match a_loc {
            Some(Loc::Reg(r)) => vec![dst_r, Reg::Rsp, r],
            _ => vec![dst_r, Reg::Rsp],
        };
        let tmp = match alloc.free_volatile_at(pos, &avoid) {
            Some(r) => r,
            None => {
                // Nothing free: borrow a register, park its value in a scratch slot and put it back once the operation has read it. Slot 1, because `with_spilled_dst` owns slot 0 and may be holding the destination's scratch right now.
                let borrowed = crate::ir::regalloc::VOLATILE
                    .iter()
                    .copied()
                    .find(|r| !avoid.contains(r))
                    .expect("the volatile set is larger than the avoid set");
                asm.mov(
                    qword_ptr(rsp + layout.scratch_slot_disp(1) as i32),
                    qreg(borrowed),
                )?;
                restore_b_holder = Some(borrowed);
                borrowed
            }
        };
        // Always a full 64-bit move: a narrower one would leave the upper half behind,
        // and `restore_b_holder` has to put the borrowed register back exactly.
        asm.mov(qreg(tmp), qreg(dst_r))?;
        b_relocated = Some(tmp);
    }

    // Get `a` into dst_r.
    match a_loc {
        Some(Loc::Reg(r)) if r != dst_r => match width {
            Width::W64 => asm.mov(qreg(dst_r), qreg(r))?,
            // A 32-bit move also carries a W16 value: the low 16 bits are the value, and every W16 consumer reads it through `wreg`, so whatever sits in bits 16..31 is never observed.
            Width::W16 | Width::W32 => asm.mov(dreg(dst_r), dreg(r))?,
            Width::W8 => asm.mov(breg(dst_r), breg(r))?,
        },
        Some(Loc::Spill(i)) => {
            let d = layout.spill_disp(i) as i32;
            match width {
                Width::W64 => asm.mov(qreg(dst_r), qword_ptr(rsp + d))?,
                Width::W16 | Width::W32 => asm.mov(dreg(dst_r), dword_ptr(rsp + d))?,
                Width::W8 => asm.mov(breg(dst_r), byte_ptr(rsp + d))?,
            }
        }
        None => emit_mov_operand_imm_or_frame(asm, dst_r, a, width, layout)?,
        _ => {} // already dst_r
    }

    // Pick a scratch for `b` that does not overlap the destination, RSP, or the
    // register currently holding `a`.
    let scratch = alloc
        .free_volatile_at(pos, &{
            match a_loc {
                Some(Loc::Reg(r)) => vec![dst_r, Reg::Rsp, r],
                _ => vec![dst_r, Reg::Rsp],
            }
        })
        .unwrap_or(if dst_r != Reg::Rax {
            Reg::Rax
        } else {
            Reg::Rcx
        });

    // Read `b` from wherever it ended up: the relocation above, or its own home.
    let read_b = |asm: &mut CodeAssembler| -> Result<Reg> {
        match b_relocated {
            Some(r) => Ok(r),
            None => ensure_reg(asm, alloc, b, pos, width, scratch, layout),
        }
    };

    // Apply the operation.
    //
    // The immediate arms need no relocation guard: `resolve` returns `None` for an
    // immediate, so `b_relocated` is only ever set when `b` is a register operand.
    match (op, b, width) {
        // Immediate forms.
        (Add, Operand::Imm(c), Width::W64) if fits_imm32(*c) => asm.add(qreg(dst_r), *c as i32)?,
        (Add, Operand::Imm(c), Width::W32) => asm.add(dreg(dst_r), *c as i32 as u32)?,
        (Add, Operand::Imm(c), Width::W16) => asm.add(wreg(dst_r), *c as u16 as i32)?,
        (Sub, Operand::Imm(c), Width::W64) if fits_imm32(*c) => asm.sub(qreg(dst_r), *c as i32)?,
        (Sub, Operand::Imm(c), Width::W32) => asm.sub(dreg(dst_r), *c as i32 as u32)?,
        (Sub, Operand::Imm(c), Width::W16) => asm.sub(wreg(dst_r), *c as u16 as i32)?,
        (And, Operand::Imm(c), Width::W64) if fits_imm32(*c) => asm.and(qreg(dst_r), *c as i32)?,
        (And, Operand::Imm(c), Width::W32) => asm.and(dreg(dst_r), *c as i32 as u32)?,
        (And, Operand::Imm(c), Width::W16) => asm.and(wreg(dst_r), *c as u16 as i32)?,
        (And, Operand::Imm(c), Width::W8) => asm.and(breg(dst_r), *c as u8 as u32)?,
        (Or, Operand::Imm(c), Width::W64) if fits_imm32(*c) => asm.or(qreg(dst_r), *c as i32)?,
        (Or, Operand::Imm(c), Width::W32) => asm.or(dreg(dst_r), *c as i32 as u32)?,
        (Or, Operand::Imm(c), Width::W16) => asm.or(wreg(dst_r), *c as u16 as i32)?,
        (Or, Operand::Imm(c), Width::W8) => asm.or(breg(dst_r), *c as u8 as u32)?,
        (Xor, Operand::Imm(c), Width::W64) if fits_imm32(*c) => asm.xor(qreg(dst_r), *c as i32)?,
        (Xor, Operand::Imm(c), Width::W32) => asm.xor(dreg(dst_r), *c as i32 as u32)?,
        (Xor, Operand::Imm(c), Width::W16) => asm.xor(wreg(dst_r), *c as u16 as i32)?,
        (Xor, Operand::Imm(c), Width::W8) => asm.xor(breg(dst_r), *c as u8 as u32)?,
        (Shl, Operand::Imm(c), _) => shift_by_imm!(asm, shl, dst_r, *c as u32, width),
        (Shr, Operand::Imm(c), _) => shift_by_imm!(asm, shr, dst_r, *c as u32, width),
        (Sar, Operand::Imm(c), _) => shift_by_imm!(asm, sar, dst_r, *c as u32, width),
        // Rotate immediates use the immediate encoding; register-count forms always
        // read CL.
        (Rol, Operand::Imm(c), _) => shift_by_imm!(asm, rol, dst_r, *c as u32, width),
        (Ror, Operand::Imm(c), _) => shift_by_imm!(asm, ror, dst_r, *c as u32, width),
        (Mul, Operand::Imm(c), Width::W64) if fits_imm32(*c) => {
            asm.imul_3(qreg(dst_r), qreg(dst_r), *c as i32)?
        }
        (Mul, Operand::Imm(c), Width::W32) => asm.imul_3(dreg(dst_r), dreg(dst_r), *c as i32)?,
        // `imul r16, r/m16, imm16` exists, and the W8 case rides the 32-bit form for
        // the reason given at the register arm: the low byte of a product depends only
        // on the low bytes of its operands.
        (Mul, Operand::Imm(c), Width::W16) => {
            asm.imul_3(wreg(dst_r), wreg(dst_r), *c as u16 as i32)?
        }
        (Mul, Operand::Imm(c), Width::W8) => {
            asm.imul_3(dreg(dst_r), dreg(dst_r), *c as u8 as i32)?
        }
        (Eq | Ult | Slt, Operand::Imm(c), Width::W64) if fits_imm32(*c) => {
            emit_cmp_imm_setcc(asm, op, dst_r, *c, width)?;
        }
        (Eq | Ult | Slt, Operand::Imm(c), Width::W32 | Width::W16 | Width::W8) => {
            emit_cmp_imm_setcc(asm, op, dst_r, *c, width)?;
        }
        (Eq | Ult | Slt, _, _) => {
            let b_r = read_b(asm)?;
            match width {
                Width::W64 => asm.cmp(qreg(dst_r), qreg(b_r))?,
                Width::W32 => asm.cmp(dreg(dst_r), dreg(b_r))?,
                Width::W16 => asm.cmp(wreg(dst_r), wreg(b_r))?,
                Width::W8 => asm.cmp(breg(dst_r), breg(b_r))?,
            }
            emit_setcc(asm, op, dst_r)?;
        }
        // Register forms: load b if needed, then operate.
        _ => {
            let b_r = read_b(asm)?;
            debug_assert!(
                b_r != dst_r || matches!(b, Operand::Imm(_)) || a == b,
                "`b` must not share `dst_r` with a different `a`: {op:?} a={a:?} b={b:?}"
            );
            match op {
                // The count has to reach CL, which the shape below cannot express.
                Shl | Shr | Sar | Rol | Ror => {
                    emit_shift_by_reg(asm, alloc, op, dst_r, b_r, width, pos, layout)?;
                }
                MulHiU | MulHiS | UDiv | URem | SDiv | SRem => {
                    unreachable!("the RDX:RAX opcodes are handled before `a` is moved into dst_r")
                }
                Eq | Ult | Slt => unreachable!("handled by the comparison arm above"),
                // Everything else is a plain two-register ALU form: look the encoding
                // up by (opcode, width) and emit it.
                _ => {
                    let codes = bin_reg_codes(op)
                        .expect("the opcodes without a table entry are matched above");
                    // `Mul` at W8 has no byte form, so it rides the dword one. Both
                    // operands widen together, which is what keeps the low byte right.
                    let (code, at) = match codes.for_width(width) {
                        Some(c) => (c, width),
                        None => (codes.d, Width::W32),
                    };
                    asm.add_instruction(iced_x86::Instruction::with2(
                        code,
                        ireg(dst_r, at),
                        ireg(b_r, at),
                    )?)?;
                }
            }
        }
    }

    // Put back a register borrowed to hold `b`. The result is in `dst_r`, which is
    // never the borrowed register, so this cannot overwrite it. `mov` leaves flags
    // alone, so a `setcc` result computed above survives.
    if let Some(r) = restore_b_holder {
        asm.mov(qreg(r), qword_ptr(rsp + layout.scratch_slot_disp(1) as i32))?;
    }
    Ok(())
}

/// The high-half multiplies and the divides, which x86 only offers on RDX:RAX. The difficulty is not the opcode but the fixed registers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_rdx_rax_op(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    dst_r: Reg,
    op: crate::ir::expr::BinOp,
    a: &Operand,
    b: &Operand,
    width: Width,
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    use crate::ir::expr::BinOp::*;

    // The divisor/multiplicand needs a register that is neither RAX nor RDX.
    let a_loc = resolve(alloc, a, pos);
    let holder = crate::ir::regalloc::VOLATILE
        .iter()
        .copied()
        .chain(crate::ir::regalloc::ALLOCATABLE.iter().copied())
        .find(|r| !matches!(r, Reg::Rax | Reg::Rdx | Reg::Rsp) && a_loc != Some(Loc::Reg(*r)))
        .expect("15 allocatable registers outnumber the four excluded here");

    let (s_rax, s_rdx, s_hold) = (
        layout.scratch_slot_disp(1) as i32,
        layout.scratch_slot_disp(2) as i32,
        layout.scratch_slot_disp(3) as i32,
    );
    asm.mov(qword_ptr(rsp + s_rax), qreg(Reg::Rax))?;
    asm.mov(qword_ptr(rsp + s_rdx), qreg(Reg::Rdx))?;
    asm.mov(qword_ptr(rsp + s_hold), qreg(holder))?;

    // `b` into `holder` first, while every register still holds what the allocation says. Doing it after `a` is loaded would read a clobbered RAX if `b` lived there.
    match resolve(alloc, b, pos) {
        Some(Loc::Reg(r)) => asm.mov(qreg(holder), qreg(r))?,
        _ => {
            let r = ensure_reg(asm, alloc, b, pos, width, holder, layout)?;
            if r != holder {
                asm.mov(qreg(holder), qreg(r))?;
            }
        }
    }

    // `a` into RAX, at the operation's width. The narrow widths matter: `mul r/m8`
    // reads AL and writes AX, so only the low bits of RAX are the operand.
    match a_loc {
        Some(Loc::Reg(r)) if r == Reg::Rax => {}
        Some(Loc::Reg(r)) => match width {
            Width::W64 => asm.mov(qreg(Reg::Rax), qreg(r))?,
            Width::W16 | Width::W32 => asm.mov(dreg(Reg::Rax), dreg(r))?,
            Width::W8 => asm.mov(breg(Reg::Rax), breg(r))?,
        },
        Some(Loc::Spill(i)) => {
            let d = layout.spill_disp(i) as i32;
            match width {
                Width::W64 => asm.mov(qreg(Reg::Rax), qword_ptr(rsp + d))?,
                Width::W16 | Width::W32 => asm.mov(dreg(Reg::Rax), dword_ptr(rsp + d))?,
                Width::W8 => asm.mov(breg(Reg::Rax), byte_ptr(rsp + d))?,
            }
        }
        _ => emit_mov_operand_imm_or_frame(asm, Reg::Rax, a, width, layout)?,
    }

    // The dividend's high half. A multiply writes RDX rather than reading it, so only
    // the divides need this.
    match op {
        UDiv | URem => match width {
            // `xor edx, edx` clears all 64 bits and is shorter than the qword form.
            Width::W64 | Width::W32 => asm.xor(dreg(Reg::Rdx), dreg(Reg::Rdx))?,
            // `div r/m16` reads DX:AX and `div r/m8` reads AX, so the 8-bit form needs
            // AH cleared rather than DX. `movzx eax, al` does that without naming AH,
            // which avoids the REX-prefix restriction on the high-byte registers.
            Width::W16 => asm.xor(dreg(Reg::Rdx), dreg(Reg::Rdx))?,
            Width::W8 => asm.movzx(dreg(Reg::Rax), breg(Reg::Rax))?,
        },
        SDiv | SRem => match width {
            Width::W64 => asm.cqo()?,
            Width::W32 => asm.cdq()?,
            Width::W16 => asm.cwd()?,
            // `idiv r/m8` reads AX, so the sign extension goes into AH: `cbw`.
            Width::W8 => asm.cbw()?,
        },
        _ => {}
    }

    match (op, width) {
        (MulHiU, Width::W64) => asm.mul(qreg(holder))?,
        (MulHiU, Width::W32) => asm.mul(dreg(holder))?,
        (MulHiU, Width::W16) => asm.mul(wreg(holder))?,
        (MulHiU, Width::W8) => asm.mul(breg(holder))?,
        (MulHiS, Width::W64) => asm.imul(qreg(holder))?,
        (MulHiS, Width::W32) => asm.imul(dreg(holder))?,
        (MulHiS, Width::W16) => asm.imul(wreg(holder))?,
        (MulHiS, Width::W8) => asm.imul(breg(holder))?,
        (UDiv | URem, Width::W64) => asm.div(qreg(holder))?,
        (UDiv | URem, Width::W32) => asm.div(dreg(holder))?,
        (UDiv | URem, Width::W16) => asm.div(wreg(holder))?,
        (UDiv | URem, Width::W8) => asm.div(breg(holder))?,
        (SDiv | SRem, Width::W64) => asm.idiv(qreg(holder))?,
        (SDiv | SRem, Width::W32) => asm.idiv(dreg(holder))?,
        (SDiv | SRem, Width::W16) => asm.idiv(wreg(holder))?,
        (SDiv | SRem, Width::W8) => asm.idiv(breg(holder))?,
        _ => unreachable!("only the RDX:RAX opcodes reach here"),
    }

    // Where the answer is. The 8-bit forms are the exception: `mul r/m8` puts the whole
    // 16-bit product in AX, so its high half is AH, and `div r/m8` leaves the remainder
    // in AH rather than DL.
    enum Src {
        Rax,
        Rdx,
        RaxHigh8,
    }
    let src = match (op, width) {
        (MulHiU | MulHiS, Width::W8) => Src::RaxHigh8,
        (MulHiU | MulHiS, _) => Src::Rdx,
        (UDiv | SDiv, _) => Src::Rax,
        (URem | SRem, Width::W8) => Src::RaxHigh8,
        (URem | SRem, _) => Src::Rdx,
        _ => unreachable!("only the RDX:RAX opcodes reach here"),
    };

    // Move the answer out before the saves are restored, since restoring overwrites both RAX and RDX.
    match src {
        Src::Rax => {
            if dst_r != Reg::Rax {
                asm.mov(qreg(dst_r), qreg(Reg::Rax))?;
            }
        }
        Src::Rdx => {
            if dst_r != Reg::Rdx {
                asm.mov(qreg(dst_r), qreg(Reg::Rdx))?;
            }
        }
        // Shift AH down into AL rather than naming AH: `shr eax, 8` needs no REX
        // reasoning and leaves the byte where every W8 consumer reads it.
        Src::RaxHigh8 => {
            asm.shr(dreg(Reg::Rax), 8u32)?;
            if dst_r != Reg::Rax {
                asm.mov(qreg(dst_r), qreg(Reg::Rax))?;
            }
        }
    }

    // Restore the saves. A register the allocation names as the result's home is left
    // alone: putting the old value back there would discard the result that was just
    // moved into it.
    if dst_r != Reg::Rax {
        asm.mov(qreg(Reg::Rax), qword_ptr(rsp + s_rax))?;
    }
    if dst_r != Reg::Rdx {
        asm.mov(qreg(Reg::Rdx), qword_ptr(rsp + s_rdx))?;
    }
    if dst_r != holder {
        asm.mov(qreg(holder), qword_ptr(rsp + s_hold))?;
    }
    Ok(())
}

/// A shift or rotate by a register count.
///
/// The count is brought to RCX with `xchg`, which needs no scratch register and is its
/// own inverse, so the swap can be undone afterwards to leave every other register as
/// the allocation describes. `xchg` does not touch flags, so a `Bin` whose flags are
/// consumed by a fused compare is unaffected.
///
/// The two interesting cases both fall out of the same pair of swaps:
///
/// - `dst_r == Rcx`: the value to shift travels to `b_r` across the first `xchg`, so
///   the shift operates on `b_r`, and the second `xchg` brings the result back to RCX.
/// - `b_r == Rcx`: the count is already in place, so no swap is needed at all.
///
/// `dst_r == b_r` is the case the swaps cannot express. Value and count are the same
/// register; `x << x`, which the allocator's handoff rule permits; and `xchg` would
/// move the value out to RCX and bring an unrelated value back in, so the shift would
/// read the wrong operand *and* the second `xchg` would discard the result. There the
/// count is copied to RCX instead, leaving the value where it is.
pub(crate) fn emit_shift_by_reg(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    op: crate::ir::expr::BinOp,
    dst_r: Reg,
    b_r: Reg,
    width: Width,
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    use crate::ir::expr::BinOp::*;

    // How the count reaches CL, and what the shift then operates on.
    enum Plan {
        /// The count is already in RCX.
        InPlace,
        /// Swap the count into RCX, shift, swap back.
        Swap { target: Reg },
        /// Copy the count into RCX, preserving whatever RCX held.
        Copy { restore: bool },
    }
    let plan = if b_r == Reg::Rcx {
        Plan::InPlace
    } else if dst_r == b_r {
        // The value being shifted is also the count, so it must stay put. RCX has to be
        // preserved unless the allocator says nothing live is in it.
        let rcx_free = alloc.free_volatile_at(pos, &[dst_r, Reg::Rsp]) == Some(Reg::Rcx);
        Plan::Copy { restore: !rcx_free }
    } else if dst_r == Reg::Rcx {
        Plan::Swap { target: b_r }
    } else {
        Plan::Swap { target: dst_r }
    };

    let target = match plan {
        Plan::InPlace | Plan::Copy { .. } => dst_r,
        Plan::Swap { target } => target,
    };

    match plan {
        Plan::InPlace => {}
        // Always the 64-bit exchange: a narrower one would leave the upper half of
        // both registers behind, and the second `xchg` has to restore `b_r` exactly.
        Plan::Swap { .. } => asm.xchg(qreg(Reg::Rcx), qreg(b_r))?,
        Plan::Copy { restore } => {
            if restore {
                // Slot 1: `with_spilled_dst` owns slot 0 and may be holding the
                // destination's scratch while this runs.
                asm.mov(
                    qword_ptr(rsp + layout.scratch_slot_disp(1) as i32),
                    qreg(Reg::Rcx),
                )?;
            }
            asm.mov(qreg(Reg::Rcx), qreg(b_r))?;
        }
    }
    match (op, width) {
        (Shl, Width::W64) => asm.shl(qreg(target), cl)?,
        (Shl, Width::W32) => asm.shl(dreg(target), cl)?,
        (Shl, Width::W16) => asm.shl(wreg(target), cl)?,
        (Shl, Width::W8) => asm.shl(breg(target), cl)?,
        (Shr, Width::W64) => asm.shr(qreg(target), cl)?,
        (Shr, Width::W32) => asm.shr(dreg(target), cl)?,
        (Shr, Width::W16) => asm.shr(wreg(target), cl)?,
        (Shr, Width::W8) => asm.shr(breg(target), cl)?,
        (Sar, Width::W64) => asm.sar(qreg(target), cl)?,
        (Sar, Width::W32) => asm.sar(dreg(target), cl)?,
        (Sar, Width::W16) => asm.sar(wreg(target), cl)?,
        (Sar, Width::W8) => asm.sar(breg(target), cl)?,
        (Rol, Width::W64) => asm.rol(qreg(target), cl)?,
        (Rol, Width::W32) => asm.rol(dreg(target), cl)?,
        (Rol, Width::W16) => asm.rol(wreg(target), cl)?,
        (Rol, Width::W8) => asm.rol(breg(target), cl)?,
        (Ror, Width::W64) => asm.ror(qreg(target), cl)?,
        (Ror, Width::W32) => asm.ror(dreg(target), cl)?,
        (Ror, Width::W16) => asm.ror(wreg(target), cl)?,
        (Ror, Width::W8) => asm.ror(breg(target), cl)?,
        _ => unreachable!("only shifts and rotates reach here"),
    }
    match plan {
        Plan::InPlace => {}
        Plan::Swap { .. } => asm.xchg(qreg(Reg::Rcx), qreg(b_r))?,
        Plan::Copy { restore } => {
            // The result is in `dst_r`, which is not RCX here (`dst_r == b_r != Rcx`),
            // so restoring RCX cannot overwrite it.
            if restore {
                asm.mov(
                    qreg(Reg::Rcx),
                    qword_ptr(rsp + layout.scratch_slot_disp(1) as i32),
                )?;
            }
        }
    }
    Ok(())
}

pub(crate) fn emit_load(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    dst_r: Reg,
    addr: &Operand,
    disp: i64,
    width: Width,
    pos: usize,
    layout: &FrameLayout,
    img: ImageRange,
) -> Result<()> {
    let d32 = disp as i32;
    match addr {
        Operand::Frame(o) => {
            let d = layout.frame_disp(*o) as i32 + d32;
            match width {
                Width::W64 => asm.mov(qreg(dst_r), qword_ptr(rsp + d))?,
                // W16 stays on the 32-bit form here, and only here. A frame slot is ours, slots are 8 bytes apart, so reading 32 bits of a 16-bit slot cannot touch a neighbour, and it avoids a partial-register write.
                Width::W16 | Width::W32 => asm.mov(dreg(dst_r), dword_ptr(rsp + d))?,
                Width::W8 => asm.mov(breg(dst_r), byte_ptr(rsp + d))?,
            }
        }
        Operand::Imm(a) => {
            // Concrete address. Three forms, cheapest first:
            let abs = (*a as i64).wrapping_add(disp);
            if img.rip_ok(abs as u64) {
                rip_load(asm, dst_r, abs as u64, width)?;
            } else if abs >= i32::MIN as i64 && abs <= i32::MAX as i64 {
                match width {
                    Width::W64 => asm.mov(qreg(dst_r), qword_ptr(abs))?,
                    Width::W32 => asm.mov(dreg(dst_r), dword_ptr(abs))?,
                    Width::W16 => asm.mov(wreg(dst_r), word_ptr(abs))?,
                    Width::W8 => asm.mov(breg(dst_r), byte_ptr(abs))?,
                }
            } else {
                emit_mov_imm(asm, dst_r, abs as u64, Width::W64, layout)?;
                match width {
                    Width::W64 => asm.mov(qreg(dst_r), qword_ptr(qreg(dst_r)))?,
                    Width::W32 => asm.mov(dreg(dst_r), dword_ptr(qreg(dst_r)))?,
                    Width::W16 => asm.mov(wreg(dst_r), word_ptr(qreg(dst_r)))?,
                    Width::W8 => asm.mov(breg(dst_r), byte_ptr(qreg(dst_r)))?,
                }
            }
        }
        _ => {
            // Computed address: load it into the dst register first, then use it.
            let addr_r = match resolve(alloc, addr, pos) {
                Some(Loc::Reg(r)) => r,
                Some(Loc::Spill(i)) => {
                    asm.mov(qreg(dst_r), qword_ptr(rsp + layout.spill_disp(i) as i32))?;
                    dst_r
                }
                None => return Ok(()),
            };
            match width {
                Width::W64 => asm.mov(qreg(dst_r), qword_ptr(qreg(addr_r) + d32))?,
                Width::W32 => asm.mov(dreg(dst_r), dword_ptr(qreg(addr_r) + d32))?,
                Width::W16 => asm.mov(wreg(dst_r), word_ptr(qreg(addr_r) + d32))?,
                Width::W8 => asm.mov(breg(dst_r), byte_ptr(qreg(addr_r) + d32))?,
            }
        }
    }
    Ok(())
}

pub(crate) fn emit_store(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    addr: &Operand,
    value: &Operand,
    disp: i64,
    width: Width,
    pos: usize,
    layout: &FrameLayout,
    img: ImageRange,
) -> Result<()> {
    // The address goes in rcx. A wide immediate value materialises through r11 and
    // an out-of-range absolute address through r10, both chosen so that neither can
    // clobber the register already holding the address.
    let addr_scratch = Reg::Rcx;

    let addr_r = match addr {
        Operand::Frame(_) | Operand::Imm(_) | Operand::OutArg(_) => None,
        _ => Some(ensure_reg(
            asm,
            alloc,
            addr,
            pos,
            Width::W64,
            addr_scratch,
            layout,
        )?),
    };
    let val_imm: Option<u64> = match value {
        Operand::Imm(c) => Some(*c),
        _ => None,
    };
    // A register borrowed to carry the value, to be put back after the store.
    let mut restore_val_holder: Option<Reg> = None;
    let val_r: Option<Reg> = if val_imm.is_some() {
        None
    } else {
        // Where the value already lives, if it is in a register.
        let home = match value {
            Operand::Entry(r) | Operand::Param(r) => alloc
                .loc_at(Item::Saved(*r), pos)
                .or_else(|| alloc.loc_at(Item::Entry(*r), pos)),
            Operand::Val(v) => alloc.loc_at(Item::Val(*v), pos),
            _ => None,
        };
        match home {
            Some(Loc::Reg(r)) => Some(r),
            // Materialize non-register values before storing them.
            _ => {
                // A scratch that cannot collide with anything this store still needs: the address register, RSP, and the two registers the macro below uses for a wide immediate (r11) and an out-of-range absolute address (r10).
                let mut avoid = vec![Reg::Rsp, Reg::R10, Reg::R11];
                if let Some(r) = addr_r {
                    avoid.push(r);
                }
                let scratch = match alloc.free_volatile_at(pos, &avoid) {
                    Some(r) => r,
                    None => {
                        // Nothing free: borrow, parking the previous contents in
                        // scratch slot 1. Slot 0 belongs to `with_spilled_dst`, which
                        // may be holding a destination's scratch right now. A store
                        // has no destination of its own, so no other slot user can be
                        // live here.
                        let borrowed = crate::ir::regalloc::VOLATILE
                            .iter()
                            .copied()
                            .find(|r| !avoid.contains(r))
                            .expect("the volatile set is larger than the avoid set");
                        asm.mov(
                            qword_ptr(rsp + layout.scratch_slot_disp(1) as i32),
                            qreg(borrowed),
                        )?;
                        restore_val_holder = Some(borrowed);
                        borrowed
                    }
                };
                // `ensure_reg` covers all of it: a frame address becomes an `lea`, a
                // spill becomes a reload, an `OutArg` becomes an `lea` off the
                // outgoing-argument area.
                Some(ensure_reg(asm, alloc, value, pos, width, scratch, layout)?)
            }
        }
    };
    macro_rules! store_to_mem {
        ($mem:expr) => {
            match (width, val_imm, val_r) {
                // A qword store takes at most a sign-extended imm32 directly, so a
                // constant in that range needs no register at all. Anything wider
                // goes through r11 rather than rax, since rax may already hold the
                // address this store's memory operand was built from.
                (Width::W64, Some(c), _)
                    if (c as i64) >= i32::MIN as i64 && (c as i64) <= i32::MAX as i64 =>
                {
                    asm.mov($mem, c as i32)?;
                }
                (Width::W64, Some(c), _) => {
                    if img.rip_ok(c) {
                        rip_lea(asm, Reg::R11, c)?;
                    } else {
                        asm.mov(r11, c as i64)?;
                    }
                    asm.mov($mem, r11)?;
                }
                (Width::W32, Some(c), _) => asm.mov($mem, c as i32 as u32)?,
                (Width::W16, Some(c), _) => asm.mov($mem, c as u16 as u32)?,
                // Byte stores explicitly truncate because iced rejects out-of-range
                // immediates.
                (Width::W8, Some(c), _) => asm.mov($mem, c as u8 as u32)?,
                (Width::W64, None, Some(r)) => asm.mov($mem, qreg(r))?,
                (Width::W32, None, Some(r)) => asm.mov($mem, dreg(r))?,
                (Width::W16, None, Some(r)) => asm.mov($mem, wreg(r))?,
                (Width::W8, None, Some(r)) => asm.mov($mem, breg(r))?,
                (w, i, r) => bail!("no store form for width {w:?} imm {i:?} reg {r:?}"),
            }
        };
    }

    let d32 = disp as i32;
    match addr {
        Operand::Frame(o) => {
            let d = layout.frame_disp(*o) as i32 + d32;
            store_to_mem!(sized(rsp + d, width));
        }
        Operand::OutArg(n) => {
            // Already measured from the RSP the call will see, so it needs none of
            // `frame_disp`'s entry-RSP correction: the displacement *is* the offset.
            // `disp` was folded to zero when the store was re-anchored.
            let d = *n as i32 + d32;
            store_to_mem!(sized(rsp + d, width));
        }
        Operand::Imm(a) => {
            let a = (*a as i64).wrapping_add(disp);
            let rip_done = if img.rip_ok(a as u64) {
                match (val_imm, val_r) {
                    (Some(c), _) => rip_store_imm(asm, a as u64, c, width)?,
                    (None, Some(r)) => {
                        rip_store_reg(asm, a as u64, r, width)?;
                        true
                    }
                    (None, None) => false,
                }
            } else {
                false
            };
            if rip_done {
                // done
            } else if a >= i32::MIN as i64 && a <= i32::MAX as i64 {
                store_to_mem!(sized(a, width));
            } else if img.rip_ok(a as u64) {
                rip_lea(asm, Reg::R10, a as u64)?;
                store_to_mem!(sized(r10, width));
            } else {
                // Absolute address outside the ±2GiB displacement range: load it
                // into a register that neither the address nor value paths use.
                asm.mov(r10, a)?;
                store_to_mem!(sized(r10, width));
            }
        }
        _ => {
            let r = match addr_r {
                Some(r) => r,
                None => return Ok(()),
            };
            store_to_mem!(sized(qreg(r) + d32, width));
        }
    }
    // Restore any register borrowed to carry the value.
    if let Some(r) = restore_val_holder {
        asm.mov(qreg(r), qword_ptr(rsp + layout.scratch_slot_disp(1) as i32))?;
    }
    Ok(())
}

/// Materialize a call's arguments into their Win64 ABI registers. This is a parallel copy, not a sequence of independent moves: an argument whose source is another argument's destination register would be destroyed if the moves were emitted naively.
pub(crate) fn emit_call_args(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    args: &[(Reg, Operand)],
    pos: usize,
    layout: &FrameLayout,
) -> Result<()> {
    // Split into register-sourced copies (need ordering) and the rest.
    let mut pending: Vec<(Reg, Reg)> = Vec::new(); // (dst, src)
    let mut direct: Vec<(Reg, &Operand)> = Vec::new();

    for (dst, op) in args {
        match resolve(alloc, op, pos) {
            Some(Loc::Reg(src)) => {
                if src != *dst {
                    pending.push((*dst, src));
                }
            }
            _ => direct.push((*dst, op)),
        }
    }

    // Emit copies whose destination is not some other pending copy's source.
    // Repeat until only cycles remain.
    while !pending.is_empty() {
        let ready = pending
            .iter()
            .position(|(d, _)| !pending.iter().any(|(_, s)| s == d));
        match ready {
            Some(i) => {
                let (d, src) = pending.remove(i);
                asm.mov(qreg(d), qreg(src))?;
            }
            None => {
                // Every remaining destination is also a source: a cycle. Swapping
                // the first pair shortens it by one without needing a temporary.
                let (d, src) = pending.remove(0);
                asm.xchg(qreg(d), qreg(src))?;
                // Whatever still wanted to read `d` now finds its value in `src`.
                for (_, s) in pending.iter_mut() {
                    if *s == d {
                        *s = src;
                    }
                }
            }
        }
    }

    // Now the immediates and memory reloads, which clobber nothing else.
    for (dst, op) in direct {
        match op {
            Operand::Imm(c) => emit_mov_imm(asm, dst, *c, Width::W64, layout)?,
            Operand::Frame(o) => {
                asm.lea(qreg(dst), qword_ptr(rsp + layout.frame_disp(*o) as i32))?
            }
            _ => {
                // Spilled: reload straight into the argument register.
                let r = ensure_reg(asm, alloc, op, pos, Width::W64, dst, layout)?;
                if r != dst {
                    asm.mov(qreg(dst), qreg(r))?;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn emit_call(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    target: &CallTarget,
    pos: usize,
    img: ImageRange,
    layout: &FrameLayout,
    redirect: &CallRedirects,
) -> Result<()> {
    match target {
        CallTarget::Direct(a) => asm.call(redirect.resolve(*a as u64))?,
        CallTarget::Import { slot } => {
            // call qword ptr [slot] — the IAT entry the real code uses.
            let a = *slot as i64;
            if img.rip_ok(*slot) {
                asm.add_instruction(Instruction::with1(Code::Call_rm64, rip_at(*slot))?)?;
            } else if a >= i32::MIN as i64 && a <= i32::MAX as i64 {
                asm.call(qword_ptr(a))?;
            } else {
                asm.mov(r11, a)?;
                asm.call(qword_ptr(r11))?;
            }
        }
        CallTarget::Indirect(op) => {
            // Prefer the allocated register and reload spilled targets through R11.
            match resolve(alloc, op, pos) {
                Some(Loc::Reg(r)) => asm.call(qreg(r))?,
                Some(Loc::Spill(i)) => {
                    // R11 is call-clobbered and never holds a value past a call
                    // site, so it is always safe to use here.
                    asm.mov(r11, qword_ptr(rsp + layout.spill_disp(i) as i32))?;
                    asm.call(r11)?
                }
                None => asm.int3()?,
            }
        }
        CallTarget::Unknown { .. } => {
            asm.int3()?;
        }
    }
    Ok(())
}

/// Emit one of the allocator's split copies: move a value from where it was to
/// where its next segment expects it.
pub(crate) fn emit_copy(
    asm: &mut CodeAssembler,
    c: &regalloc::Copy_,
    layout: &FrameLayout,
) -> Result<()> {
    match (c.from, c.to) {
        (Loc::Reg(a), Loc::Reg(b)) if a == b => {}
        (Loc::Reg(a), Loc::Reg(b)) => asm.mov(qreg(b), qreg(a))?,
        (Loc::Reg(a), Loc::Spill(i)) => {
            asm.mov(qword_ptr(rsp + layout.spill_disp(i) as i32), qreg(a))?
        }
        (Loc::Spill(i), Loc::Reg(b)) => {
            asm.mov(qreg(b), qword_ptr(rsp + layout.spill_disp(i) as i32))?
        }
        (Loc::Spill(i), Loc::Spill(j)) if i == j => {}
        (Loc::Spill(i), Loc::Spill(j)) => {
            // No stack-to-stack move exists; bounce through a scratch register.
            // R11 is call-clobbered, so it never holds a value that must survive.
            asm.mov(r11, qword_ptr(rsp + layout.spill_disp(i) as i32))?;
            asm.mov(qword_ptr(rsp + layout.spill_disp(j) as i32), r11)?;
        }
    }
    Ok(())
}
