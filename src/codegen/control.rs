use super::*;

/// Remove any move targeting RSP. Emitted code keeps RSP at a fixed offset from the function's entry throughout the body: the prologue reserves the frame and the epilogue restores it, so emitting a boundary move that changes RSP would misalign every subsequent frame-slot access.
pub(crate) fn strip_rsp(moves: &[Move]) -> Vec<Move> {
    moves
        .iter()
        .filter(|m| {
            !matches!(m,
                Move::Set { dst, .. } | Move::Imm { dst, .. } | Move::Frame { dst, .. }
                    if *dst == Reg::Rsp
            )
        })
        .cloned()
        .collect()
}

pub(crate) fn emit_moves(
    asm: &mut CodeAssembler,
    moves: &[Move],
    layout: &FrameLayout,
) -> Result<()> {
    emit_moves_opts(asm, moves, layout, false)
}

/// Registers a boundary move writes.
///
/// A swap writes both halves. Used to decide whether the moves would destroy the
/// value a branch is about to compare.
pub(crate) fn move_writes(m: &Move) -> Vec<Reg> {
    match m {
        Move::Set { dst, .. } | Move::Imm { dst, .. } | Move::Frame { dst, .. } => vec![*dst],
        Move::Swap { a, b } => vec![*a, *b],
    }
}

/// Registers a boundary move reads.
///
/// A `Frame` computes an address and an `Imm` materializes a constant, so neither
/// reads anything. Used to keep a reordered comparison's scratch register off a
/// value the moves still have to read.
pub(crate) fn move_reads(m: &Move) -> Vec<Reg> {
    match m {
        Move::Set {
            src: Loc::Reg(r), ..
        } => vec![*r],
        Move::Set {
            src: Loc::Spill(_), ..
        }
        | Move::Imm { .. }
        | Move::Frame { .. } => Vec::new(),
        Move::Swap { a, b } => vec![*a, *b],
    }
}

/// Emit boundary moves, optionally preserving flags. `keep_flags` matters only when the moves are emitted *after* a comparison, which happens when they would otherwise overwrite the register that comparison reads.
pub(crate) fn emit_moves_opts(
    asm: &mut CodeAssembler,
    moves: &[Move],
    layout: &FrameLayout,
    keep_flags: bool,
) -> Result<()> {
    for m in moves {
        match m {
            Move::Set {
                dst,
                src: Loc::Reg(r),
            } => {
                asm.mov(qreg(*dst), qreg(*r))?;
            }
            Move::Set {
                dst,
                src: Loc::Spill(i),
            } => {
                asm.mov(qreg(*dst), qword_ptr(rsp + layout.spill_disp(*i) as i32))?;
            }
            // `mov r32, 0` rather than `xor`: three bytes longer, and it does not
            // touch the flags a pending Jcc is about to read.
            Move::Imm { dst, value } if keep_flags && *value == 0 => {
                asm.mov(dreg(*dst), 0u32)?;
            }
            Move::Imm { dst, value } => {
                emit_mov_imm(asm, *dst, *value, Width::W64, layout)?;
            }
            Move::Frame { dst, offset } => {
                asm.lea(
                    qreg(*dst),
                    qword_ptr(rsp + layout.frame_disp(*offset) as i32),
                )?;
            }
            Move::Swap { a, b } => {
                asm.xchg(qreg(*a), qreg(*b))?;
            }
        }
    }
    Ok(())
}

/// Two scratch registers for a comparison, avoiding `avoid`. The pair was fixed at RAX/RCX.
pub(crate) fn cmp_scratch(avoid: &[Reg]) -> (Reg, Reg) {
    let mut free = regalloc::VOLATILE
        .iter()
        .copied()
        .filter(|r| !avoid.contains(r));
    // Falling back to the original pair keeps this total. `avoid` comes from the
    // boundary moves, which never name seven registers in practice.
    let a = free.next().unwrap_or(Reg::Rax);
    let b = free.next().unwrap_or(Reg::Rcx);
    (a, b)
}

pub(crate) fn emit_cmp(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    a: &Operand,
    b: &Operand,
    width: Width,
    def_pos: usize,
    layout: &FrameLayout,
    avoid: &[Reg],
) -> Result<()> {
    let (scratch, scratch_b) = cmp_scratch(avoid);
    let a_r = ensure_reg(asm, alloc, a, def_pos, width, scratch, layout)?;
    match b {
        Operand::Imm(0) => match width {
            Width::W64 => asm.test(qreg(a_r), qreg(a_r))?,
            Width::W32 => asm.test(dreg(a_r), dreg(a_r))?,
            Width::W16 => asm.test(wreg(a_r), wreg(a_r))?,
            Width::W8 => asm.test(breg(a_r), breg(a_r))?,
        },
        Operand::Imm(c) if !(matches!(width, Width::W64) && !fits_imm32(*c)) => match width {
            Width::W64 => asm.cmp(qreg(a_r), *c as i32)?,
            Width::W32 => asm.cmp(dreg(a_r), *c as i32 as u32)?,
            Width::W16 => asm.cmp(wreg(a_r), *c as u16 as i32)?,
            Width::W8 => asm.cmp(breg(a_r), *c as u8 as u32)?,
        },
        _ => {
            let b_r = ensure_reg(asm, alloc, b, def_pos, width, scratch_b, layout)?;
            match width {
                Width::W64 => asm.cmp(qreg(a_r), qreg(b_r))?,
                Width::W32 => asm.cmp(dreg(a_r), dreg(b_r))?,
                Width::W16 => asm.cmp(wreg(a_r), wreg(b_r))?,
                Width::W8 => asm.cmp(breg(a_r), breg(b_r))?,
            }
        }
    }
    Ok(())
}

/// Reload the callee-saved registers before the frame goes away. Emitted after the boundary moves, not before.
pub(crate) fn emit_restores(asm: &mut CodeAssembler, layout: &FrameLayout) -> Result<()> {
    for &r in layout.saves() {
        let d = layout
            .save_disp(r)
            .expect("save slot for a register in saves()");
        asm.mov(qreg(r), qword_ptr(rsp + d as i32))?;
    }
    Ok(())
}

/// Emit the block's exit. A jump to `next` is omitted so control falls through.
pub(crate) fn emit_terminator(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    b: &SchedBlock,
    moves: &[Move],
    label_of: &HashMap<BlockId, CodeLabel>,
    layout: &FrameLayout,
    next: Option<BlockId>,
    redirect: &CallRedirects,
) -> Result<()> {
    // Resolve a target to a label, dropping it when it is the fallthrough.
    let jump_lbl = |tgt: &BlockId| -> Option<CodeLabel> {
        if next == Some(*tgt) {
            return None;
        }
        label_of.get(tgt).copied()
    };
    match &b.terminator {
        Terminator::Return { .. } => {
            // Only ABI return registers are observable after the epilogue. RSP is
            // adjusted by the frame teardown and callee-saved registers are restored.
            const RET_REGS: [Reg; 2] = [Reg::Rax, Reg::Rdx];
            let live_out: Vec<_> = moves
                .iter()
                .filter(|m| match m {
                    Move::Set { dst, .. } | Move::Imm { dst, .. } | Move::Frame { dst, .. } => {
                        RET_REGS.contains(dst)
                    }
                    // A swap writes both halves, so it survives only if a return
                    // register needs it. Restricting to the return registers means a
                    // swap can no longer be part of a cycle worth breaking.
                    Move::Swap { a, b } => RET_REGS.contains(a) || RET_REGS.contains(b),
                })
                .cloned()
                .collect();
            emit_moves(asm, &live_out, layout)?;
            emit_restores(asm, layout)?;
            if layout.frame_size > 0 {
                asm.add(rsp, layout.frame_size as i32)?;
            }
            asm.ret()?;
        }
        Terminator::Jump(tgt) | Terminator::Backedge { target: tgt } => {
            emit_moves(asm, &strip_rsp(moves), layout)?;
            if let Some(lbl) = jump_lbl(tgt) {
                asm.jmp(lbl)?;
            }
        }
        Terminator::Adopted { taken, not_taken } => {
            // The concrete pass recovered both edges but the symbolic pass could not re-derive the predicate. Attempt the same branch emission as a normal `Branch`: if `fused_cmp` or a `control` value is present, emit the comparison and a conditional jump.
            let moves = strip_rsp(moves);
            if b.control.is_some() || b.fused_cmp.is_some() {
                let read = branch_reads(alloc, b, layout);
                let clobbered = moves
                    .iter()
                    .flat_map(move_writes)
                    .any(|w| read.contains(&w));
                let taken_lbl = label_of.get(taken).copied();
                let not_taken_lbl = jump_lbl(not_taken);
                if clobbered {
                    let avoid: Vec<Reg> = moves.iter().flat_map(move_reads).collect();
                    emit_branch_cmp(asm, alloc, b, layout, &avoid)?;
                    emit_moves_opts(asm, &moves, layout, true)?;
                    emit_branch_jump(asm, b, taken_lbl, not_taken_lbl)?;
                } else {
                    emit_moves(asm, &moves, layout)?;
                    emit_branch_jcc(asm, alloc, b, taken_lbl, not_taken_lbl, layout)?;
                }
            } else {
                bail!(
                    "Adopted branch at {:?} has no recoverable predicate; \
                     retaining original trampoline",
                    b.id
                );
            }
        }
        Terminator::TailCall { target } => {
            // The tail-callee returns straight to our caller, so the registers
            // must already be restored when the jump is taken.
            emit_moves(asm, &strip_rsp(moves), layout)?;
            emit_restores(asm, layout)?;
            if layout.frame_size > 0 {
                asm.add(rsp, layout.frame_size as i32)?;
            }
            asm.jmp(redirect.resolve(*target as u64))?;
        }
        Terminator::Branch {
            taken, not_taken, ..
        } => {
            let moves = strip_rsp(moves);
            // Emit the comparison before boundary moves when those moves overwrite a
            // branch operand. Boundary moves preserve flags.
            let read = branch_reads(alloc, b, layout);
            let clobbered = moves
                .iter()
                .flat_map(move_writes)
                .any(|w| read.contains(&w));
            if clobbered {
                let avoid: Vec<Reg> = moves.iter().flat_map(move_reads).collect();
                let taken_lbl = label_of.get(taken).copied();
                let not_taken_lbl = jump_lbl(not_taken);
                emit_branch_cmp(asm, alloc, b, layout, &avoid)?;
                emit_moves_opts(asm, &moves, layout, true)?;
                emit_branch_jump(asm, b, taken_lbl, not_taken_lbl)?;
            } else {
                emit_moves(asm, &moves, layout)?;
                // Only the not-taken edge may fall through; the conditional jump
                // always names the taken target.
                let taken_lbl = label_of.get(taken).copied();
                let not_taken_lbl = jump_lbl(not_taken);
                emit_branch_jcc(asm, alloc, b, taken_lbl, not_taken_lbl, layout)?;
            }
        }
        Terminator::Switch { targets, .. } => {
            emit_moves(asm, &strip_rsp(moves), layout)?;
            // Emit a comparison chain for each case but the last. The selector is a 0-based index into `targets`, materialized in `b.control`.
            if let (2.., Some(ctrl_op)) = (targets.len(), &b.control) {
                // Find the register holding the selector.
                let end = b.instrs.len();
                if let Some(Loc::Reg(sel_r)) = resolve(alloc, ctrl_op, end) {
                    let sel_w = control_width(b, ctrl_op);
                    // For each target except the last: cmp sel, i; je target.
                    for (i, tgt) in targets[..targets.len() - 1].iter().enumerate() {
                        match sel_w {
                            Width::W64 => asm.cmp(qreg(sel_r), i as i32)?,
                            Width::W32 => asm.cmp(dreg(sel_r), i as i32)?,
                            Width::W16 => asm.cmp(wreg(sel_r), i as i32)?,
                            Width::W8 => asm.cmp(breg(sel_r), i as i32)?,
                        }
                        if let Some(lbl) = label_of.get(tgt).copied() {
                            asm.je(lbl)?;
                        }
                    }
                    // Last target: unconditional jump (or fall through).
                    let last = targets.last().unwrap();
                    if let Some(lbl) = jump_lbl(last) {
                        asm.jmp(lbl)?;
                    }
                } else {
                    // Selector in a spill slot: materialise into a scratch register.
                    // R11 is caller-saved and never holds a value the allocator cares
                    // about at a terminator, so it is safe to clobber here.
                    let scratch = Reg::R11;
                    if let Some(Loc::Spill(i)) = resolve(alloc, ctrl_op, end) {
                        asm.mov(r11, qword_ptr(rsp + layout.spill_disp(i) as i32))?;
                        let sel_w = control_width(b, ctrl_op);
                        for (idx, tgt) in targets[..targets.len() - 1].iter().enumerate() {
                            match sel_w {
                                Width::W64 => asm.cmp(qreg(scratch), idx as i32)?,
                                Width::W32 => asm.cmp(dreg(scratch), idx as i32)?,
                                Width::W16 => asm.cmp(wreg(scratch), idx as i32)?,
                                Width::W8 => asm.cmp(breg(scratch), idx as i32)?,
                            }
                            if let Some(lbl) = label_of.get(tgt).copied() {
                                asm.je(lbl)?;
                            }
                        }
                        let last = targets.last().unwrap();
                        if let Some(lbl) = jump_lbl(last) {
                            asm.jmp(lbl)?;
                        }
                    } else if let Some(tgt) = targets.first() {
                        if let Some(lbl) = jump_lbl(tgt) {
                            asm.jmp(lbl)?;
                        }
                    }
                }
            } else if let Some(tgt) = targets.first() {
                if let Some(lbl) = jump_lbl(tgt) {
                    asm.jmp(lbl)?;
                }
            }
        }
        Terminator::AdoptedSwitch { targets } => {
            // Use a recovered selector when available; otherwise retain the original
            // function instead of choosing an arbitrary edge.
            if b.control.is_some() {
                emit_moves(asm, &strip_rsp(moves), layout)?;
                // Re-use the same comparison-chain logic as `Terminator::Switch`.
                // (fall through to the Switch arm would require a different match
                // shape, so duplicate the dispatch inline.)
                let end = b.instrs.len();
                if let Some(ctrl_op) = &b.control {
                    if let Some(Loc::Reg(sel_r)) = resolve(alloc, ctrl_op, end) {
                        let sel_w = control_width(b, ctrl_op);
                        for (i, tgt) in targets[..targets.len().saturating_sub(1)]
                            .iter()
                            .enumerate()
                        {
                            match sel_w {
                                Width::W64 => asm.cmp(qreg(sel_r), i as i32)?,
                                Width::W32 => asm.cmp(dreg(sel_r), i as i32)?,
                                Width::W16 => asm.cmp(wreg(sel_r), i as i32)?,
                                Width::W8 => asm.cmp(breg(sel_r), i as i32)?,
                            }
                            if let Some(lbl) = label_of.get(tgt).copied() {
                                asm.je(lbl)?;
                            }
                        }
                        if let Some(last) = targets.last() {
                            if let Some(lbl) = jump_lbl(last) {
                                asm.jmp(lbl)?;
                            }
                        }
                    } else if let Some(tgt) = targets.first() {
                        if let Some(lbl) = jump_lbl(tgt) {
                            asm.jmp(lbl)?;
                        }
                    }
                }
            } else {
                bail!(
                    "AdoptedSwitch at {:?} has no recoverable selector; \
                     retaining original trampoline",
                    b.id
                );
            }
        }
        Terminator::AdoptedReturn => {
            // The concrete pass proved this block returns, but the symbolic
            // pass could not derive the return value expression.  Emit the
            // standard epilogue so the caller's frame is restored correctly;
            // whatever the ABI return registers already hold is the best the
            // analysis could recover.
            emit_restores(asm, layout)?;
            if layout.frame_size > 0 {
                asm.add(rsp, layout.frame_size as i32)?;
            }
            asm.ret()?;
        }
        _ => {
            // Unresolved: INT3 is an explicit crash rather than falling off
            // the function into whatever follows in memory.
            asm.int3()?;
        }
    }
    Ok(())
}

/// Registers the branch reads at the end of `b`.
///
/// Empty when the condition is not in a register: a comparison operand that has to be
/// computed into scratch cannot be clobbered by a move, since nothing holds it yet.
pub(crate) fn branch_reads(alloc: &Alloc, b: &SchedBlock, layout: &FrameLayout) -> Vec<Reg> {
    let _ = layout;
    let end = b.instrs.len();
    let mut out = Vec::new();
    if b.control.is_none() {
        return out;
    }
    match &b.fused_cmp {
        Some(fc) => {
            for o in [&fc.a, &fc.b] {
                if let Some(Loc::Reg(r)) = resolve(alloc, o, end) {
                    out.push(r);
                }
            }
        }
        None => {
            if let Some(ctrl) = &b.control {
                if let Some(Loc::Reg(r)) = resolve(alloc, ctrl, end) {
                    out.push(r);
                }
            }
        }
    }
    out
}

/// Emit only the flag-setting half of a branch, leaving the Jcc to `emit_branch_jump`.
pub(crate) fn emit_branch_cmp(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    b: &SchedBlock,
    layout: &FrameLayout,
    avoid: &[Reg],
) -> Result<()> {
    let end = b.instrs.len();
    if b.control.is_none() {
        return Ok(());
    }
    match &b.fused_cmp {
        Some(fc) => emit_cmp(asm, alloc, &fc.a, &fc.b, fc.width, end, layout, avoid)?,
        None => {
            // `branch_reads` only reported a conflict because the control value is in
            // a register, so this resolves.
            if let Some(ctrl) = &b.control {
                if let Some(Loc::Reg(r)) = resolve(alloc, ctrl, end) {
                    emit_test_self(asm, r, control_width(b, ctrl))?;
                }
            }
        }
    }
    Ok(())
}

/// Emit the Jcc and fallthrough jump for a branch whose comparison already ran.
pub(crate) fn emit_branch_jump(
    asm: &mut CodeAssembler,
    b: &SchedBlock,
    taken_lbl: Option<CodeLabel>,
    not_taken_lbl: Option<CodeLabel>,
) -> Result<()> {
    if b.control.is_some() {
        if let Some(lbl) = taken_lbl {
            match &b.fused_cmp {
                Some(fc) => {
                    use crate::ir::expr::BinOp::{Eq, Slt, Ult};
                    match fc.op {
                        Eq => asm.je(lbl)?,
                        Ult => asm.jb(lbl)?,
                        Slt => asm.jl(lbl)?,
                        other => bail!("comparison {other:?} is not a branch condition"),
                    }
                }
                None => asm.jne(lbl)?,
            }
        }
    }
    if let Some(lbl) = not_taken_lbl {
        asm.jmp(lbl)?;
    }
    Ok(())
}

pub(crate) fn emit_branch_jcc(
    asm: &mut CodeAssembler,
    alloc: &Alloc,
    b: &SchedBlock,
    taken_lbl: Option<CodeLabel>,
    not_taken_lbl: Option<CodeLabel>,
    layout: &FrameLayout,
) -> Result<()> {
    let end = b.instrs.len();
    if let Some(ctrl) = &b.control {
        // The comparison, if any, was recorded by dead-code elimination when it removed the instruction that defined it.
        if let Some(fc) = &b.fused_cmp {
            emit_cmp(asm, alloc, &fc.a, &fc.b, fc.width, end, layout, &[])?;
            use crate::ir::expr::BinOp::{Eq, Slt, Ult};
            if let Some(lbl) = taken_lbl {
                match fc.op {
                    Eq => asm.je(lbl)?,
                    Ult => asm.jb(lbl)?,
                    // Signed, so SF/OF rather than CF. Falling through to `jne`
                    // here would test ZF and take the branch on any inequality.
                    Slt => asm.jl(lbl)?,
                    other => bail!("comparison {other:?} is not a branch condition"),
                }
            }
        } else {
            // Generic: test the control value and jump if non-zero.
            let ctrl_r = match resolve(alloc, ctrl, end) {
                Some(Loc::Reg(r)) => r,
                _ => {
                    if let Some(lbl) = taken_lbl {
                        asm.jmp(lbl)?;
                    }
                    return Ok(());
                }
            };
            emit_test_self(asm, ctrl_r, control_width(b, ctrl))?;
            if let Some(lbl) = taken_lbl {
                asm.jne(lbl)?;
            }
        }
    }
    if let Some(lbl) = not_taken_lbl {
        asm.jmp(lbl)?;
    }
    Ok(())
}
