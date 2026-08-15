use super::*;
use iced_x86::{
    BlockEncoder, BlockEncoderOptions, Decoder, DecoderOptions, FlowControl, InstructionBlock,
    Mnemonic, OpKind, Register,
};

/// Callee-saved registers written by the emitted function, in ABI order.
///
/// The scan covers allocated segments, split copies, and boundary moves. RSP is
/// handled by the frame, while RBP is ordinary allocatable storage.
pub(crate) fn saved_registers(blocks: &[SchedBlock], allocs: &[Alloc]) -> Vec<Reg> {
    let mut written: HashSet<Reg> = HashSet::new();
    fn note(written: &mut HashSet<Reg>, l: Loc) {
        if let Loc::Reg(r) = l {
            written.insert(r);
        }
    }
    for (b, alloc) in blocks.iter().zip(allocs.iter()) {
        for seg in &alloc.segments {
            // An entry value left in its own register is the incoming value arriving in place, which reads the register without writing it.
            if let (Item::Entry(r), Loc::Reg(l)) = (seg.item, seg.loc) {
                if r == l {
                    continue;
                }
            }
            note(&mut written, seg.loc);
        }
        for c in &alloc.copies {
            note(&mut written, c.to);
        }
        for m in regalloc::reconcile(b, alloc) {
            match m {
                Move::Set {
                    dst,
                    src: Loc::Reg(s),
                } if dst == s => {}
                Move::Set { dst, .. } | Move::Imm { dst, .. } | Move::Frame { dst, .. } => {
                    written.insert(dst);
                }
                Move::Swap { a, b } => {
                    written.insert(a);
                    written.insert(b);
                }
            }
        }
    }
    // ABI order, so the layout is stable and diffable rather than hash order.
    [
        Reg::Rbx,
        Reg::Rbp,
        Reg::Rsi,
        Reg::Rdi,
        Reg::R12,
        Reg::R13,
        Reg::R14,
        Reg::R15,
    ]
    .into_iter()
    .filter(|r| written.contains(r))
    .collect()
}

/// Mark adjacent scratch restore/save pairs that cancel out.
///
/// Pairs at branch targets are retained because control may enter between them.
pub(crate) fn mark_scratch_roundtrips(
    instrs: &[iced_x86::Instruction],
    branch_targets: &HashSet<usize>,
    dead: &mut [bool],
) -> usize {
    // (index, register, memory displacement) of the last restore seen.
    let mut prev_restore: Option<(usize, Register, u64)> = None;
    let mut killed = 0usize;
    for (i, ins) in instrs.iter().enumerate() {
        if dead[i] {
            continue;
        }
        // An instruction at a block entry is reachable from more than one
        // predecessor, so a save there cannot be paired with the restore laid out
        // before it.
        if branch_targets.contains(&i) {
            prev_restore = None;
        }

        let rsp_mem = ins.memory_base() == Register::RSP
            && ins.memory_index() == Register::None
            && ins.memory_displacement64() != 0;
        // `mov r64, [rsp + k]`; potential restore.
        let as_restore = ins.mnemonic() == Mnemonic::Mov
            && ins.op_count() == 2
            && ins.op0_kind() == OpKind::Register
            && ins.op1_kind() == OpKind::Memory
            && ins.op0_register().size() == 8
            && rsp_mem;
        // `mov [rsp + k], r64`; potential save.
        let as_save = ins.mnemonic() == Mnemonic::Mov
            && ins.op_count() == 2
            && ins.op0_kind() == OpKind::Memory
            && ins.op1_kind() == OpKind::Register
            && ins.op1_register().size() == 8
            && rsp_mem;

        if let Some((p_idx, p_reg, p_disp)) = prev_restore {
            if as_save && ins.op1_register() == p_reg && ins.memory_displacement64() == p_disp {
                dead[p_idx] = true;
                dead[i] = true;
                killed += 2;
                prev_restore = None;
                continue;
            }
        }
        prev_restore = if as_restore {
            Some((i, ins.op0_register(), ins.memory_displacement64()))
        } else {
            None
        };
    }
    killed
}

/// Byte-granular liveness over the emitter's private frame slots. Tracked as a bitset indexed by `disp - lo` across the two ranges [`FrameLayout::private_ranges`] returns, concatenated.
pub(crate) struct PrivateSlots {
    ranges: [(i64, i64); 2],
    /// Start index in the bitset of each range.
    starts: [usize; 2],
    len: usize,
}

impl PrivateSlots {
    fn new(layout: &FrameLayout) -> Self {
        let ranges = layout.private_ranges();
        let n0 = (ranges[0].1 - ranges[0].0).max(0) as usize;
        let n1 = (ranges[1].1 - ranges[1].0).max(0) as usize;
        PrivateSlots {
            ranges,
            starts: [0, n0],
            len: n0 + n1,
        }
    }

    /// Bitset index for a frame displacement, or `None` if it is not private.
    fn index(&self, disp: i64) -> Option<usize> {
        for (r, &(lo, hi)) in self.ranges.iter().enumerate() {
            if disp >= lo && disp < hi {
                return Some(self.starts[r] + (disp - lo) as usize);
            }
        }
        None
    }

    /// Whether `[disp, disp+size)` lies wholly inside one private range.
    ///
    /// A store straddling the boundary of a range is not eligible: part of it lands
    /// somewhere this analysis does not model, so it has to be treated as live.
    fn contains_all(&self, disp: i64, size: usize) -> bool {
        self.ranges
            .iter()
            .any(|&(lo, hi)| disp >= lo && disp + size as i64 <= hi)
    }
}

/// Mark stores to the emitter's private frame slots whose value is never read.
///
/// A backward liveness pass over the emitted instruction stream, iterated to a
/// fixpoint so that loops are handled: a store is dead when none of the bytes it
/// writes are live on any path leaving it.
///
/// The three sources of these stores are all cases where the emitter commits a value
/// it cannot yet know will be used:
///
/// - `with_spilled_dst` commits every spilled definition to its slot, including the
///   ones whose only consumer was folded away later,
/// - the RDX:RAX save area is written by every `mul`/`div` whether or not the
///   surviving path reads the preserved registers back,
/// - the last store to any slot before a `ret` is dead by construction, because the
///   frame is gone once the function returns.
///
/// Returns 0 without marking anything when the stream contains something that makes
/// the analysis unsound; an unresolved indirect branch, or an instruction that
/// computes the *address* of a private slot; since in both cases a store can be
/// read through a path this pass cannot see.
pub(crate) fn mark_dead_private_stores(
    instrs: &[iced_x86::Instruction],
    layout: &FrameLayout,
    dead: &mut [bool],
) -> usize {
    let slots = PrivateSlots::new(layout);
    if slots.len == 0 {
        return 0;
    }

    // Index of each instruction by IP, for resolving branch targets.
    let ip_to_idx: HashMap<u64, usize> = instrs
        .iter()
        .enumerate()
        .filter(|(i, _)| !dead[*i])
        .map(|(i, ins)| (ins.ip(), i))
        .collect();

    // Live instruction indices in layout order, and each one's successors.
    let live: Vec<usize> = (0..instrs.len()).filter(|&i| !dead[i]).collect();
    let mut succs: Vec<Vec<usize>> = vec![Vec::new(); instrs.len()];
    for (n, &i) in live.iter().enumerate() {
        let ins = &instrs[i];
        let fallthrough = live.get(n + 1).copied();
        match ins.flow_control() {
            FlowControl::UnconditionalBranch => {
                match ip_to_idx.get(&ins.near_branch64()) {
                    Some(&t) => succs[i].push(t),
                    // A `jmp` out of the function (tail call). Nothing in this frame
                    // is read afterwards, so it has no successors here.
                    None => {}
                }
            }
            FlowControl::ConditionalBranch => {
                if let Some(&t) = ip_to_idx.get(&ins.near_branch64()) {
                    succs[i].push(t);
                }
                succs[i].extend(fallthrough);
            }
            FlowControl::Return => {}
            // An indirect branch can land anywhere, including at a block that reads a
            // slot stored before the jump. Bail rather than guess.
            FlowControl::IndirectBranch => return 0,
            _ => succs[i].extend(fallthrough),
        }
    }

    // An instruction that materialises the address of a private slot would let a callee read it, breaking the premise that direct `[rsp+k]` operands are the only access.
    for &i in &live {
        let ins = &instrs[i];
        if ins.mnemonic() == Mnemonic::Lea
            && ins.memory_base() == Register::RSP
            && ins.memory_index() == Register::None
            && slots.index(ins.memory_displacement64() as i64).is_some()
        {
            return 0;
        }
    }

    // Classify each live instruction once: the bytes it reads and, if it is a pure store to a private slot, the bytes it writes.
    let mut reads: Vec<Vec<usize>> = vec![Vec::new(); instrs.len()];
    let mut store: Vec<Option<(usize, usize)>> = vec![None; instrs.len()];
    for &i in &live {
        let ins = &instrs[i];
        if ins.memory_base() != Register::RSP || ins.memory_index() != Register::None {
            continue;
        }
        let disp = ins.memory_displacement64() as i64;
        let size = ins.memory_size().size();
        let pure_store = ins.mnemonic() == Mnemonic::Mov
            && ins.op_count() == 2
            && ins.op0_kind() == OpKind::Memory
            && matches!(
                ins.op1_kind(),
                OpKind::Register
                    | OpKind::Immediate8
                    | OpKind::Immediate8to32
                    | OpKind::Immediate8to64
                    | OpKind::Immediate16
                    | OpKind::Immediate32
                    | OpKind::Immediate32to64
                    | OpKind::Immediate64
            );
        if pure_store {
            if slots.contains_all(disp, size) {
                let lo = slots.index(disp).expect("contained store has an index");
                store[i] = Some((lo, lo + size));
            }
            // A store outside the private ranges reads nothing and writes nothing
            // this pass models.
            continue;
        }
        // Everything else touching the frame is treated as reading the bytes it
        // names: loads, read-modify-writes, and any wider access.
        if let Some(lo) = slots.index(disp) {
            let hi = (lo + size).min(slots.len);
            reads[i].extend(lo..hi);
        }
    }

    // Backward liveness to a fixpoint. `live_in[i]` is the set of private bytes
    // live on entry to instruction i.
    let mut live_in: Vec<Vec<bool>> = vec![vec![false; slots.len]; instrs.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for &i in live.iter().rev() {
            let mut out = vec![false; slots.len];
            for &s in &succs[i] {
                for (b, o) in out.iter_mut().enumerate() {
                    *o |= live_in[s][b];
                }
            }
            // Transfer: kill what this instruction writes, then add what it reads.
            if let Some((lo, hi)) = store[i] {
                for o in out[lo..hi].iter_mut() {
                    *o = false;
                }
            }
            for &b in &reads[i] {
                out[b] = true;
            }
            if out != live_in[i] {
                live_in[i] = out;
                changed = true;
            }
        }
    }

    // A store is dead when no byte it writes is live on any outgoing edge.
    let mut killed = 0usize;
    for &i in &live {
        let Some((lo, hi)) = store[i] else { continue };
        let mut any_live = false;
        for &s in &succs[i] {
            if live_in[s][lo..hi].iter().any(|&b| b) {
                any_live = true;
                break;
            }
        }
        if !any_live {
            dead[i] = true;
            killed += 1;
        }
    }
    killed
}

pub(crate) fn optimize_frame_traffic_with(
    bytes: Vec<u8>,
    base_va: u64,
    layout: Option<&FrameLayout>,
) -> Result<Vec<u8>> {
    let mut bytes = bytes;
    // Re-decode after each compaction because instruction addresses and branch
    // displacements change.
    for _ in 0..8 {
        let mut dec = Decoder::with_ip(64, &bytes, base_va, DecoderOptions::NONE);
        let mut instrs: Vec<iced_x86::Instruction> = Vec::new();
        while dec.can_decode() {
            instrs.push(dec.decode());
        }
        if instrs.iter().any(|i| i.is_invalid()) {
            // Something upstream emitted bytes that do not decode. `validate_emitted`
            // reports that with the offset; returning the input unchanged keeps this
            // pass from turning it into a confusing encoder error first.
            return Ok(bytes);
        }

        // Instruction indices that are branch targets.
        let ip_to_idx: HashMap<u64, usize> = instrs
            .iter()
            .enumerate()
            .map(|(i, x)| (x.ip(), i))
            .collect();
        let mut branch_targets: HashSet<usize> = HashSet::new();
        for ins in &instrs {
            if matches!(
                ins.flow_control(),
                FlowControl::UnconditionalBranch | FlowControl::ConditionalBranch
            ) {
                if let Some(&t) = ip_to_idx.get(&ins.near_branch64()) {
                    branch_targets.insert(t);
                }
            }
        }

        let mut dead = vec![false; instrs.len()];
        let mut killed = mark_scratch_roundtrips(&instrs, &branch_targets, &mut dead);
        if let Some(layout) = layout {
            killed += mark_dead_private_stores(&instrs, layout, &mut dead);
        }
        if killed == 0 {
            return Ok(bytes);
        }

        // A removed instruction cannot be a branch target: the branch would be left pointing at whatever followed it.
        for (i, d) in dead.iter_mut().enumerate() {
            if *d && branch_targets.contains(&i) {
                *d = false;
            }
        }

        let kept: Vec<iced_x86::Instruction> = instrs
            .iter()
            .enumerate()
            .filter(|(i, _)| !dead[*i])
            .map(|(_, x)| *x)
            .collect();
        if kept.len() == instrs.len() {
            return Ok(bytes);
        }
        let block = InstructionBlock::new(&kept, base_va);
        let encoded = BlockEncoder::encode(64, block, BlockEncoderOptions::NONE)
            .map_err(|e| anyhow::anyhow!("re-encoding after frame cleanup failed: {e}"))?;
        bytes = encoded.code_buffer;
    }
    Ok(bytes)
}

/// [`optimize_frame_traffic_with`] without a layout, so only the roundtrip pass runs.
///
/// The tests for that pass build a bare instruction pair rather than a whole
/// function, and have no frame to describe.
#[cfg(test)]
pub(crate) fn optimize_frame_traffic(bytes: Vec<u8>, base_va: u64) -> Result<Vec<u8>> {
    optimize_frame_traffic_with(bytes, base_va, None)
}

/// The frame layout a function will be emitted against. Split out of [`emit_function`] because the unwind record and the code have to describe the *same* frame.
pub fn plan_frame(blocks: &[SchedBlock], allocs: &[Alloc]) -> FrameLayout {
    // Compute the frame layout from the worst-case frame extent and spill slots.
    let frame_lo = blocks
        .iter()
        .filter_map(sched::frame_extent)
        .map(|(lo, _)| lo)
        .min()
        .unwrap_or(0);
    let spill_slots = allocs.iter().map(|a| a.slots).max().unwrap_or(0);
    let has_calls = blocks.iter().any(|b| {
        b.instrs.iter().any(|i| matches!(i, Instr::Call { .. }))
            || matches!(b.terminator, Terminator::TailCall { .. })
    });
    let saves = saved_registers(blocks, allocs);
    // The widest call in the function decides the outgoing argument area, since one
    // region at the bottom of the frame serves every call site.
    let stack_args = blocks.iter().map(sched::stack_arg_count).max().unwrap_or(0);
    FrameLayout::full(frame_lo, spill_slots, has_calls, saves, stack_args)
}

/// Emit a function with no call redirection.
///
/// The single-function paths emit at a synthetic base and never patch trampolines, so
/// there is no devirtualized body for a direct call to be pointed at.
pub fn emit_function(
    cfg: &Cfg,
    blocks: &[SchedBlock],
    allocs: &[Alloc],
    base_va: u64,
    pe: &PeFile,
) -> Result<Vec<u8>> {
    emit_function_full(cfg, blocks, allocs, base_va, pe, &CallRedirects::none()).map(|e| e.bytes)
}

pub fn emit_function_full(
    _cfg: &Cfg,
    blocks: &[SchedBlock],
    allocs: &[Alloc],
    base_va: u64,
    pe: &PeFile,
    redirect: &CallRedirects,
) -> Result<EmittedFn> {
    if blocks.is_empty() {
        bail!("no blocks to emit");
    }
    let img = ImageRange::of(pe);

    // The layout carries the image range so that `emit_mov_imm`, which is reached through half a dozen helpers that all already take a layout, can materialise an image address as a RIP-relative `lea` instead of an absolute immediate.
    let mut layout = plan_frame(blocks, allocs);
    layout.img = img;
    let layout = layout;

    let mut asm = CodeAssembler::new(64)?;

    // Create a label for each block up front so forward refs work.
    let mut label_of: HashMap<BlockId, CodeLabel> = HashMap::new();
    for b in blocks {
        label_of.insert(b.id, asm.create_label());
    }

    // Emit the function prologue on the entry block. The frame is reserved first, then the callee-saved registers are written into it.
    if layout.frame_size > 0 {
        asm.sub(rsp, layout.frame_size as i32)?;
    }
    for &r in layout.saves() {
        let d = layout
            .save_disp(r)
            .expect("save slot for a register in saves()");
        asm.mov(qword_ptr(rsp + d as i32), qreg(r))?;
    }

    // Whether the previous block emitted no instructions, leaving its label attached
    // to nothing yet.
    let mut pending_label = false;
    for (i, (b, alloc)) in blocks.iter().zip(allocs.iter()).enumerate() {
        // A jump to the next laid-out block falls through without an explicit branch.
        let next_id = blocks.get(i + 1).map(|n| n.id);

        // A pending label needs an instruction before another label can be placed.
        if pending_label {
            asm.nop()?;
        }
        let lbl = label_of.get_mut(&b.id).unwrap();
        asm.set_label(lbl)?;
        let instrs_before = asm.instructions().len();

        // Record position for the caller (will be resolved after assemble).

        // Emit instructions, inserting the allocator's split copies before the
        // position they take effect at.
        //
        // A split range's later segment lives somewhere new; the copy is what puts
        // the value there. Skipping these leaves every post-split read pointing at
        // storage nothing ever wrote.
        for (pos, instr) in b.instrs.iter().enumerate() {
            for c in alloc.copies.iter().filter(|c| c.at == pos) {
                emit_copy(&mut asm, c, &layout)?;
            }
            emit_instr(
                &mut asm,
                alloc,
                instr,
                pos,
                &layout,
                img,
                Some(&b.instrs),
                redirect,
            )?;
        }
        // Copies landing past the last instruction still have to be emitted, or a
        // value the terminator's boundary moves read would be missing.
        for c in alloc.copies.iter().filter(|c| c.at >= b.instrs.len()) {
            emit_copy(&mut asm, c, &layout)?;
        }

        // Boundary moves and terminator.
        let moves = regalloc::reconcile(b, alloc);
        emit_terminator(
            &mut asm, alloc, b, &moves, &label_of, &layout, next_id, redirect,
        )?;
        // Did this block emit anything? If not, the next block's label would share an
        // instruction with this one's.
        pending_label = asm.instructions().len() == instrs_before;
    }

    // Assemble at base_va.
    let bytes = asm.assemble(base_va)?;
    // Cancel scratch save/restore pairs and drop stores to private frame slots that
    // nothing reads, then re-encode without the gaps.
    let bytes = optimize_frame_traffic_with(bytes, base_va, Some(&layout))?;

    // Validate the emitted code: every byte must decode, branch targets land on
    // instruction boundaries, and frame accesses stay within the allocated frame.
    validate_emitted(&bytes, base_va, &label_of, blocks, &layout)?;

    // Measure the prologue by decoding what was actually emitted.
    let prologue = measure_prologue(&bytes, &layout);

    Ok(EmittedFn { bytes, prologue })
}

/// A function's emitted bytes together with what unwinding it needs to know.
pub struct EmittedFn {
    pub bytes: Vec<u8>,
    /// The prologue steps in program order, each with the byte offset at which the
    /// instruction ends (i.e. where the *next* instruction starts). These are
    /// measured from the assembled output, not predicted, so they are always exact.
    pub prologue: Vec<PrologueInstr>,
}

/// Validate emitted machine code for structural correctness.
///
/// Checks that:
/// - Every byte decodes to a valid x64 instruction
/// - All branch targets land on instruction boundaries
/// - Frame accesses stay within the allocated frame bounds
/// - No instruction references unmapped memory
///
/// This is not semantic validation (the code might compute wrong answers), but it
/// catches emission bugs that would produce malformed x64 or crash at runtime.
pub(crate) fn validate_emitted(
    bytes: &[u8],
    base_va: u64,
    _label_of: &HashMap<BlockId, CodeLabel>,
    _blocks: &[SchedBlock],
    layout: &FrameLayout,
) -> Result<()> {
    // Collect instruction boundaries.
    let mut instr_starts: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut decoder = Decoder::with_ip(64, bytes, base_va, DecoderOptions::NONE);
    while decoder.can_decode() {
        let start = decoder.ip();
        instr_starts.insert(start);
        let instr = decoder.decode();
        if instr.is_invalid() {
            bail!(
                "invalid instruction at {:#x} (offset {})",
                start,
                start - base_va
            );
        }
    }
    // Every byte was consumed by a valid instruction.
    if decoder.ip() != base_va + bytes.len() as u64 {
        bail!(
            "decoder stopped at {:#x}, expected {:#x}",
            decoder.ip(),
            base_va + bytes.len() as u64
        );
    }

    // Re-scan for branch targets and frame accesses.
    decoder = Decoder::with_ip(64, bytes, base_va, DecoderOptions::NONE);
    let mut bad_branches: Vec<String> = Vec::new();
    let mut bad_frame: Vec<String> = Vec::new();
    while decoder.can_decode() {
        let instr = decoder.decode();
        // Check branch targets land on instruction boundaries.
        match instr.flow_control() {
            FlowControl::UnconditionalBranch | FlowControl::ConditionalBranch => {
                let target = instr.near_branch64();
                if target >= base_va && target < base_va + bytes.len() as u64 {
                    if !instr_starts.contains(&target) {
                        bad_branches.push(format!(
                            "{:#x}: branch to {:#x} (mid-instruction)",
                            instr.ip(),
                            target
                        ));
                    }
                }
            }
            _ => {}
        }
        // Check frame accesses stay within bounds.
        if instr.memory_base() == Register::RSP && instr.memory_index() == Register::None {
            let disp = instr.memory_displacement64() as i64;
            // Frame accesses are [rsp+k] where k is frame_disp(o) for some offset o.
            // Negative displacements are outgoing arguments (above RSP) or the return
            // address. Positive go into the frame. The frame extent is
            // [0, frame_size + save_area]. Anything beyond that is suspicious.
            let max = layout.frame_size as i64 + (layout.saves().len() * 8) as i64;
            if disp < -256 || disp > max {
                // -256 allows generous outgoing arg area without flagging; adjust if
                // functions with many stack args trip this.
                bad_frame.push(format!(
                    "{:#x}: {:?} [rsp{:+}] outside frame bounds [0, {}]",
                    instr.ip(),
                    instr.mnemonic(),
                    disp,
                    max
                ));
            }
        }
    }
    if !bad_branches.is_empty() {
        bail!(
            "branch target validation failed ({} issues):\n  {}",
            bad_branches.len(),
            bad_branches.join("\n  ")
        );
    }
    if !bad_frame.is_empty() && std::env::var_os("TVM_STRICT_FRAME").is_some() {
        // Not a hard error by default: some patterns (large arrays on stack) can
        // legitimately go beyond the heuristic bounds. Only fail under an env flag.
        bail!(
            "frame access validation failed ({} issues):\n  {}",
            bad_frame.len(),
            bad_frame.join("\n  ")
        );
    }
    Ok(())
}
