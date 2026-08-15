//! Scheduling: expression DAGs plus ordered effects into linear three-address code.
//!
//! Recovery leaves each block as a set of expression DAGs (one per live guest
//! register, plus one per memory effect) and a list of effects in program order.
//! Emitting machine code needs a sequence, so the DAGs have to be linearized.
//!
//! The original instruction sequence is neither recoverable nor needed. Any
//! sequence computing the same values and performing the same effects in the same
//! order is a correct devirtualization, so this is ordinary SSA scheduling rather
//! than reverse engineering.
//!
//! Two properties are load-bearing:
//!
//! - **Shared subexpressions are emitted once.** The arena is hash-consed, so a
//!   value reachable from several roots is one node, and a post-order walk with a
//!   visited set naturally gives it one definition. This is where the common
//!   subexpression elimination comes from; no separate pass is needed.
//! - **Effects keep their program order.** Loads and stores can alias, so
//!   reordering them is not sound in general. Their *address and value* DAGs are
//!   scheduled as late as possible but always before the effect that consumes
//!   them.

use crate::ir::expr::{Arena, BinOp, Op, Ref, Reg, UnOp, Width};
use crate::ir::lift::Event;
use crate::ir::{Block, BlockId, Cfg, Terminator};
use std::collections::{HashMap, HashSet};

/// A scheduled value: the result of one operation.
///
/// Numbered separately from `Ref` because scheduling assigns each computed value a
/// position in the sequence, and several `Ref`s can map to the same slot when a
/// value is rematerialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValId(pub u32);

impl std::fmt::Display for ValId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// An operand to a scheduled instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// A previously scheduled value.
    Val(ValId),
    /// An immediate.
    Imm(u64),
    /// A guest register as it was on entry to this block, i.e. an SSA parameter.
    /// Register allocation resolves these against the block's incoming edges.
    Param(Reg),
    /// A guest register on entry to the whole function.
    Entry(Reg),
    /// A stack location, as a signed offset from RSP on entry to the function. Recovery pins RSP to a concrete `stack_base`, so every stack reference comes out as an absolute address.
    Frame(i64),
    OutArg(u32),
}

/// The memory operand of a boxed instruction, addressed like any other operand.
#[derive(Debug, Clone)]
pub struct BoxedMemOp {
    pub addr: Operand,
    /// Access size in bytes. 16 for SSE, which no `Width` covers.
    pub bytes: u32,
    pub writes: bool,
}

/// One scheduled operation, in three-address form.
#[derive(Debug, Clone)]
pub enum Instr {
    /// `dst = a op b`
    /// `width` is the width of the *result*. For a comparison that is `W8`, because
    /// the result is a boolean, so it does not describe what the comparison measures.
    /// `operand_width` carries that, read from the arena where both are known.
    Bin {
        dst: ValId,
        op: BinOp,
        a: Operand,
        b: Operand,
        width: Width,
        operand_width: Width,
    },
    /// `dst = op a`
    Un {
        dst: ValId,
        op: UnOp,
        a: Operand,
        width: Width,
    },
    /// `dst = zero/sign-extend or truncate a`
    Cast {
        dst: ValId,
        kind: CastKind,
        a: Operand,
        from: Width,
        to: Width,
    },
    /// `dst = cond ? a : b`
    Select {
        dst: ValId,
        cond: Operand,
        a: Operand,
        b: Operand,
        width: Width,
    },
    /// `dst = [addr + disp]`
    ///
    /// `disp` exists so a constant offset can live in the x86 memory operand
    /// instead of a separate `Add`. See [`fold_addressing`].
    Load {
        dst: ValId,
        addr: Operand,
        disp: i64,
        width: Width,
    },
    /// `[addr + disp] = value`
    Store {
        addr: Operand,
        value: Operand,
        disp: i64,
        width: Width,
    },
    /// A call to `target`, or an indirect call through a computed value.
    ///
    /// `args` names the values the Win64 argument registers must hold at the call.
    /// Codegen materializes these into RCX/RDX/R8/R9 immediately before the call.
    Call {
        target: CallTarget,
        args: Vec<(Reg, Operand)>,
    },
    /// An instruction the lifter does not model, re-emitted verbatim.
    Boxed {
        site: u64,
        text: String,
        /// Original encoding, re-decoded and re-encoded by codegen.
        bytes: Vec<u8>,
        /// The memory operand, with its address as a schedulable operand.
        mem: Option<BoxedMemOp>,
        /// The values the instruction's *implicit* register reads must find in place. Same role as `Call::args`, and materialized the same way.
        uses: Vec<(Reg, Operand)>,
    },
    /// `dst` is an unmodelled result, e.g. an `af_undef` flag or a register a callee clobbered. Nothing is emitted for it; the definition exists so uses have something to refer to.
    Opaque {
        dst: ValId,
        tag: &'static str,
        width: Width,
        at: Option<Reg>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CastKind {
    Zext,
    Sext,
    Trunc,
}

#[derive(Debug, Clone, Copy)]
pub enum CallTarget {
    Direct(u64),
    Indirect(Operand),
    /// A call through an import address table slot. The slot address is the static
    /// answer: the qword in it is an unbound hint the loader overwrites, so there is
    /// no callee address in the file to name.
    Import {
        slot: u64,
    },
    /// The callee could not be recovered. Kept explicit rather than guessed, since
    /// a fabricated target would produce code that calls the wrong address.
    Unknown {
        site: u64,
    },
}

/// A block lowered to a linear instruction sequence.
#[derive(Debug, Clone)]
pub struct SchedBlock {
    pub id: crate::ir::BlockId,
    pub instrs: Vec<Instr>,
    /// Guest register values leaving the block, as scheduled operands. This is what
    /// feeds the successors' parameters.
    pub exits: Vec<(Reg, Operand)>,
    /// Registers left holding whatever a callee returned. The emitted `call` already
    /// establishes these, so they carry no instructions.
    pub callee_set: Vec<Reg>,
    /// The branch predicate or switch selector, once scheduled.
    pub control: Option<Operand>,
    /// A comparison folded into the terminator's conditional jump. When `control` is produced by a single `Eq`/`Ult`/`Slt` that nothing else reads, the branch can test the operands directly and the comparison never needs to become a value.
    pub fused_cmp: Option<FusedCmp>,
    pub terminator: Terminator,
}

/// A comparison lifted out of the instruction stream into the branch.
#[derive(Debug, Clone)]
pub struct FusedCmp {
    pub a: Operand,
    pub b: Operand,
    pub op: BinOp,
    pub width: Width,
}

/// How far from the pinned RSP a constant may be and still count as a stack address.
pub const FRAME_WINDOW: u64 = 0x8000;

/// Linearize one block.
pub struct Scheduler<'a> {
    arena: &'a Arena,
    stack_base: u64,
    block: crate::ir::expr::BlockRef,
    /// Values already scheduled, so a shared subexpression is emitted once.
    placed: HashMap<Ref, Operand>,
    instrs: Vec<Instr>,
    next: u32,
}

impl<'a> Scheduler<'a> {
    pub fn new(arena: &'a Arena, block: crate::ir::expr::BlockRef, stack_base: u64) -> Self {
        Self {
            arena,
            block,
            stack_base,
            placed: HashMap::new(),
            instrs: Vec::new(),
            next: 0,
        }
    }

    fn fresh(&mut self) -> ValId {
        let v = ValId(self.next);
        self.next += 1;
        v
    }

    /// Schedule `r` and everything it depends on, returning how to refer to it. Iterative rather than recursive: these DAGs nest hundreds of levels deep after MBA rewriting, which is the same reason `Arena::graft` is iterative.
    fn frame_offset(&self, c: u64) -> Option<i64> {
        let delta = c.wrapping_sub(self.stack_base) as i64;
        (delta.unsigned_abs() <= FRAME_WINDOW).then_some(delta)
    }

    pub fn value(&mut self, r: Ref) -> Operand {
        if let Some(&o) = self.placed.get(&r) {
            return o;
        }

        // Post-order so operands are scheduled before the operation using them.
        let mut order: Vec<Ref> = Vec::new();
        let mut queued: HashSet<Ref> = HashSet::new();
        let mut stack = vec![(r, false)];
        while let Some((cur, expanded)) = stack.pop() {
            if self.placed.contains_key(&cur) {
                continue;
            }
            if expanded {
                order.push(cur);
                continue;
            }
            if !queued.insert(cur) {
                continue;
            }
            stack.push((cur, true));
            for c in self.arena.children(cur) {
                if !self.placed.contains_key(&c) {
                    stack.push((c, false));
                }
            }
        }

        for cur in order {
            if self.placed.contains_key(&cur) {
                continue;
            }
            let o = self.emit(cur);
            self.placed.insert(cur, o);
        }
        self.placed[&r]
    }

    /// Emit the single operation for `r`, whose operands are already scheduled.
    fn emit(&mut self, r: Ref) -> Operand {
        let width = self.arena.width(r);
        match *self.arena.op(r) {
            // Leaves cost nothing: they name a value that already exists.
            // A constant inside the stack window is a pinned stack address, not a
            // literal the original code contained. Keeping it relative to entry RSP is
            // both emittable and the actual fact recovery established.
            Op::Const(c) => match self.frame_offset(c) {
                Some(off) => Operand::Frame(off),
                None => Operand::Imm(c),
            },
            Op::InitReg(reg) => Operand::Entry(reg),
            Op::Param(owner, reg) => {
                debug_assert_eq!(
                    owner, self.block,
                    "scheduling a foreign block's parameter; SSA scoping is violated"
                );
                Operand::Param(reg)
            }
            Op::Opaque(tag, _) => {
                // A `call_ret` reaching here was not produced by any call in this block: the call path below places its opaque directly, so a block-local return value is already in `placed` and never arrives here.
                if tag == "call_ret" {
                    return Operand::Entry(Reg::Rax);
                }
                let dst = self.fresh();
                // No architectural home: this is a lazily materialized opaque, reached
                // because some reader asked for a value nothing defines. Codegen emits
                // nothing and leaves the register alone.
                self.instrs.push(Instr::Opaque {
                    dst,
                    tag,
                    width,
                    at: None,
                });
                Operand::Val(dst)
            }
            Op::Load(addr, _) => {
                let addr = self.placed[&addr];
                let dst = self.fresh();
                self.instrs.push(Instr::Load {
                    dst,
                    addr,
                    disp: 0,
                    width,
                });
                Operand::Val(dst)
            }
            Op::Bin(op, a, b) => {
                // Read the operand width before `a`/`b` are replaced by scheduled
                // operands, which no longer carry one.
                let operand_width = self.arena.width(a).max(self.arena.width(b));
                let (a, b) = (self.placed[&a], self.placed[&b]);
                let dst = self.fresh();
                // `x + (-stack_addr)` is a subtraction of a stack address, and the negated constant is as unemittable as the address itself.
                if let (crate::ir::expr::BinOp::Add, Operand::Imm(c)) = (op, b) {
                    if let Some(off) = self.frame_offset((c as i64).wrapping_neg() as u64) {
                        self.instrs.push(Instr::Bin {
                            dst,
                            op: crate::ir::expr::BinOp::Sub,
                            a,
                            b: Operand::Frame(off),
                            width,
                            operand_width,
                        });
                        return Operand::Val(dst);
                    }
                }
                self.instrs.push(Instr::Bin {
                    dst,
                    op,
                    a,
                    b,
                    width,
                    operand_width,
                });
                Operand::Val(dst)
            }
            Op::Un(op, a) => {
                let a = self.placed[&a];
                let dst = self.fresh();
                self.instrs.push(Instr::Un { dst, op, a, width });
                Operand::Val(dst)
            }
            Op::Zext(a) | Op::Sext(a) | Op::Trunc(a) => {
                let kind = match self.arena.op(r) {
                    Op::Zext(_) => CastKind::Zext,
                    Op::Sext(_) => CastKind::Sext,
                    _ => CastKind::Trunc,
                };
                let from = self.arena.width(a);
                let a = self.placed[&a];
                let dst = self.fresh();
                self.instrs.push(Instr::Cast {
                    dst,
                    kind,
                    a,
                    from,
                    to: width,
                });
                Operand::Val(dst)
            }
            Op::Select(c, a, b) => {
                let (c, a, b) = (self.placed[&c], self.placed[&a], self.placed[&b]);
                let dst = self.fresh();
                self.instrs.push(Instr::Select {
                    dst,
                    cond: c,
                    a,
                    b,
                    width,
                });
                Operand::Val(dst)
            }
        }
    }
}

/// Schedule every block of a recovered function.
pub fn schedule(cfg: &Cfg) -> Vec<SchedBlock> {
    schedule_counted(cfg).0
}

/// As [`schedule`], also reporting how many instructions DCE removed.
pub fn schedule_counted(cfg: &Cfg) -> (Vec<SchedBlock>, usize) {
    let mut out: Vec<SchedBlock> = cfg
        .blocks
        .iter()
        .map(|b| {
            let mut sb = schedule_block(cfg, b);
            fold_addressing(&mut sb);
            sb
        })
        .collect();
    // Before pruning: this changes which registers count as read.
    rename_entry_reads(cfg, &mut out);
    prune_dead_exits(cfg, &mut out);
    let mut removed = 0usize;
    // Whole-function, so it must be computed before any block is rewritten: a
    // pointer can be minted in one block and used in another.
    let escapes = analyze_frame_escapes(&out);
    // Also whole-function, and for the same reason. Runs before the per-block passes
    // so the value chains it orphans are collected by the DCE below rather than
    // surviving to codegen.
    let reads = analyze_frame_reads(&out);
    drop_write_only_frame_slots(&mut out, &reads, &escapes);
    for sb in &mut out {
        // Before DSE: narrowing can leave a store whose remaining byte is itself
        // overwritten later, which DSE is then free to delete outright.
        narrow_partially_dead_stores(sb);
        removed += eliminate_dead_with(sb, &escapes);
    }
    merge_identical_returns(&mut out);
    thread_empty_jumps(&mut out);
    let out = drop_unreachable(out);
    (out, removed)
}

/// The blocks a terminator can transfer control to.
fn successors_of(t: &crate::ir::Terminator) -> Vec<crate::ir::BlockId> {
    match t {
        Terminator::Jump(x) | Terminator::Backedge { target: x } => vec![*x],
        Terminator::Branch {
            taken, not_taken, ..
        }
        | Terminator::Adopted { taken, not_taken } => vec![*taken, *not_taken],
        Terminator::Switch { targets, .. } | Terminator::AdoptedSwitch { targets } => {
            targets.clone()
        }
        Terminator::Return { .. }
        | Terminator::TailCall { .. }
        | Terminator::Unresolved { .. }
        | Terminator::AdoptedReturn => Vec::new(),
    }
}

/// Drop blocks no longer reachable from the entry. [`thread_empty_jumps`] retargets edges around blocks that do nothing, which leaves those blocks with no predecessors. The entry block is always kept, since it is the function's only entry point.
fn drop_unreachable(blocks: Vec<SchedBlock>) -> Vec<SchedBlock> {
    let Some(entry) = blocks.first().map(|b| b.id) else {
        return blocks;
    };
    let by_id: HashMap<_, _> = blocks.iter().map(|b| (b.id, b)).collect();
    let mut seen = HashSet::new();
    let mut stack = vec![entry];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        if let Some(b) = by_id.get(&id) {
            stack.extend(
                successors_of(&b.terminator)
                    .into_iter()
                    .filter(|s| !seen.contains(s)),
            );
        }
    }
    blocks
        .into_iter()
        .filter(|b| seen.contains(&b.id))
        .collect()
}

fn thread_empty_jumps(blocks: &mut [SchedBlock]) {
    /// Where this block forwards to, if standing in for it changes nothing.
    fn forwards_to(sb: &SchedBlock) -> Option<BlockId> {
        if !sb.instrs.is_empty() {
            return None;
        }
        // An exit is a no-op if the value is already in its target register. RSP
        // exits are also ignored because codegen keeps RSP frame-relative.
        let exits_are_noops = sb.exits.iter().all(|(reg, op)| {
            *reg == Reg::Rsp || matches!(op, Operand::Entry(r) | Operand::Param(r) if r == reg)
        });
        if !exits_are_noops {
            return None;
        }
        match sb.terminator {
            Terminator::Jump(t) => Some(t),
            _ => None,
        }
    }

    let target_of: HashMap<BlockId, BlockId> = blocks
        .iter()
        .filter_map(|sb| forwards_to(sb).map(|t| (sb.id, t)))
        .collect();

    // Follow chains to their end. The bound prevents malformed empty-block cycles
    // from spinning here.
    let resolve = |mut id: BlockId| -> BlockId {
        for _ in 0..target_of.len() {
            match target_of.get(&id) {
                // Stop rather than step onto a block that forwards to itself.
                Some(&next) if next != id => id = next,
                _ => break,
            }
        }
        id
    };

    for sb in blocks.iter_mut() {
        match &mut sb.terminator {
            Terminator::Jump(t) | Terminator::Backedge { target: t } => *t = resolve(*t),
            Terminator::Branch {
                taken, not_taken, ..
            }
            | Terminator::Adopted { taken, not_taken } => {
                *taken = resolve(*taken);
                *not_taken = resolve(*not_taken);
            }
            Terminator::Switch { targets, .. } | Terminator::AdoptedSwitch { targets } => {
                for t in targets.iter_mut() {
                    *t = resolve(*t);
                }
            }
            Terminator::Return { .. }
            | Terminator::TailCall { .. }
            | Terminator::Unresolved { .. }
            | Terminator::AdoptedReturn => {}
        }
    }
}

fn narrow_partially_dead_stores(b: &mut SchedBlock) -> usize {
    // Byte address -> index of the instruction that most recently wrote it.
    let mut last: HashMap<i64, usize> = HashMap::new();
    // Store index -> the byte addresses of it that some reader can still see.
    let mut survives: HashMap<usize, HashSet<i64>> = HashMap::new();
    // Store index -> (base address, width in bytes).
    let mut spans: Vec<(usize, i64, i64)> = Vec::new();

    let freeze = |last: &mut HashMap<i64, usize>,
                  survives: &mut HashMap<usize, HashSet<i64>>,
                  range: Option<(i64, i64)>| {
        match range {
            // A read of [a, a+n) freezes exactly those bytes.
            Some((a, n)) => {
                for k in 0..n {
                    if let Some(&w) = last.get(&(a + k)) {
                        survives.entry(w).or_default().insert(a + k);
                    }
                }
            }
            // A barrier freezes everything currently visible.
            None => {
                for (&addr, &w) in last.iter() {
                    survives.entry(w).or_default().insert(addr);
                }
                last.clear();
            }
        }
    };

    for (i, ins) in b.instrs.iter().enumerate() {
        match ins {
            Instr::Store {
                addr: Operand::Frame(off),
                disp,
                width,
                ..
            } => {
                let a = off + disp;
                let n = width.bytes() as i64;
                spans.push((i, a, n));
                for k in 0..n {
                    last.insert(a + k, i);
                }
            }
            Instr::Load {
                addr: Operand::Frame(off),
                disp,
                width,
                ..
            } => {
                freeze(
                    &mut last,
                    &mut survives,
                    Some((off + disp, width.bytes() as i64)),
                );
            }
            // Any store we cannot resolve to a frame byte, and any call or boxed
            // instruction, may read or alias anything.
            Instr::Store { .. } | Instr::Load { .. } | Instr::Call { .. } | Instr::Boxed { .. } => {
                freeze(&mut last, &mut survives, None);
            }
            _ => {}
        }
    }
    // Whatever is still visible when the block ends is readable by a successor.
    freeze(&mut last, &mut survives, None);

    let mut narrowed = 0;
    for (i, a, n) in spans {
        let Some(live) = survives.get(&i) else {
            continue;
        };
        if live.is_empty() || live.len() as i64 >= n {
            continue;
        }
        // Only a prefix anchored at the store's own base can be re-expressed as a
        // narrower store at the same address.
        let keep = live.len() as i64;
        if !(0..keep).all(|k| live.contains(&(a + k))) {
            continue;
        }
        // And only widths the ISA has.
        let Some(w) = Width::from_bytes(keep as u32) else {
            continue;
        };
        if let Instr::Store { width, value, .. } = &mut b.instrs[i] {
            *width = w;
            // The value keeps its own width; codegen writes only the low bytes.
            // Truncate an immediate so the emitted constant reads as the key byte
            // rather than a wide value the store silently cuts down.
            if let Operand::Imm(v) = value {
                let mask = match w {
                    Width::W8 => 0xff,
                    Width::W16 => 0xffff,
                    Width::W32 => 0xffff_ffff,
                    Width::W64 => u64::MAX,
                };
                *value = Operand::Imm(*v & mask);
            }
            narrowed += 1;
        }
    }
    narrowed
}

fn merge_identical_returns(blocks: &mut [SchedBlock]) {
    let mut canonical: HashMap<String, BlockId> = HashMap::new();
    // Never merge the entry block away: index 0 is the function's entry and codegen
    // emits the prologue on it, so it has to survive even if a later return happens
    // to look identical.
    let mut replacement: HashMap<BlockId, BlockId> = HashMap::new();
    for (i, sb) in blocks.iter().enumerate() {
        if i == 0 || !matches!(sb.terminator, Terminator::Return { .. }) {
            continue;
        }
        let key = format!("{:?}|{:?}|{:?}", sb.instrs, sb.exits, sb.terminator);
        match canonical.get(&key) {
            Some(&keep) => {
                replacement.insert(sb.id, keep);
            }
            None => {
                canonical.insert(key, sb.id);
            }
        }
    }
    if replacement.is_empty() {
        return;
    }

    // One hop is enough: a replacement always maps to a block that is itself
    // canonical, so no chain can form.
    let redirect = |id: BlockId| -> BlockId { *replacement.get(&id).unwrap_or(&id) };

    for sb in blocks.iter_mut() {
        match &mut sb.terminator {
            Terminator::Jump(t) | Terminator::Backedge { target: t } => *t = redirect(*t),
            Terminator::Branch {
                taken, not_taken, ..
            }
            | Terminator::Adopted { taken, not_taken } => {
                *taken = redirect(*taken);
                *not_taken = redirect(*not_taken);
            }
            Terminator::Switch { targets, .. } | Terminator::AdoptedSwitch { targets } => {
                for t in targets.iter_mut() {
                    *t = redirect(*t);
                }
            }
            Terminator::Return { .. }
            | Terminator::TailCall { .. }
            | Terminator::Unresolved { .. }
            | Terminator::AdoptedReturn => {}
        }
    }
}

/// How many low bits of `q`'s function-entry value the expression `v` preserves, or `None` if it is not that value at all.
fn preserved_entry_bits(
    a: &crate::ir::expr::Arena,
    v: crate::ir::expr::Ref,
    q: Reg,
) -> Option<u32> {
    match *a.op(v) {
        Op::InitReg(r) if r == q => Some(64),
        // `x & (2^n - 1)`, the shape a 32-bit move leaves behind.
        Op::Bin(BinOp::And, x, m) => {
            let mask = a.as_const(m)?;
            let bits = mask.trailing_ones();
            if mask != u64::MAX >> (64 - bits) {
                return None;
            }
            Some(preserved_entry_bits(a, x, q)?.min(bits))
        }
        Op::Trunc(x) => Some(preserved_entry_bits(a, x, q)?.min(a.width(v).bits())),
        // Zero-extending a truncation keeps the low bits and says nothing new.
        Op::Zext(x) => preserved_entry_bits(a, x, q),
        _ => None,
    }
}

/// The widest read of each guest register's entry value in this block. A narrowed home is only a valid substitute for reads that fit inside it, so the rename needs to know how wide the widest read is.
fn entry_read_widths(
    sb: &SchedBlock,
    b: &crate::ir::Block,
    arena: &crate::ir::expr::Arena,
) -> HashMap<Reg, u32> {
    let mut out: HashMap<Reg, u32> = HashMap::new();
    let mut note = |o: &Operand, bits: u32| {
        if let Operand::Entry(r) | Operand::Param(r) = o {
            let e = out.entry(*r).or_insert(0);
            *e = (*e).max(bits);
        }
    };
    for i in &sb.instrs {
        match i {
            Instr::Bin {
                op,
                a,
                b,
                operand_width,
                ..
            } => {
                // A mask observes only the bits it keeps.
                let masked = |x: &Operand, m: &Operand| -> Option<u32> {
                    if *op != BinOp::And {
                        return None;
                    }
                    let Operand::Imm(mask) = m else { return None };
                    let bits = mask.trailing_ones();
                    // Only a low-bit mask names a prefix of the value.
                    (bits > 0 && *mask == u64::MAX >> (64 - bits))
                        .then(|| bits.min(operand_width.bits()))
                        .map(|w| {
                            let _ = x;
                            w
                        })
                };
                note(a, masked(a, b).unwrap_or(operand_width.bits()));
                note(b, masked(b, a).unwrap_or(operand_width.bits()));
            }
            Instr::Un { a, width, .. } => note(a, width.bits()),
            // A truncation reads only the bits it keeps, whatever its source width claims. Charging it the source width made a 32-bit home unable to answer `trunc32(rcx)`, which is precisely the read the narrowed save exists to serve.
            Instr::Cast {
                a, kind, from, to, ..
            } => note(
                a,
                match kind {
                    CastKind::Trunc => to.bits(),
                    CastKind::Zext | CastKind::Sext => from.bits(),
                },
            ),
            Instr::Select {
                cond, a, b, width, ..
            } => {
                note(cond, 8);
                note(a, width.bits());
                note(b, width.bits());
            }
            // An address is always consumed at full width, whatever the access width.
            Instr::Load { addr, .. } => note(addr, 64),
            Instr::Store {
                addr, value, width, ..
            } => {
                note(addr, 64);
                note(value, width.bits());
            }
            // Argument registers and an indirect target are full-width reads.
            Instr::Call { target, args } => {
                if let CallTarget::Indirect(t) = target {
                    note(t, 64);
                }
                for (_, x) in args {
                    note(x, 64);
                }
            }
            // A boxed instruction is re-emitted from its own encoding, so nothing here
            // can narrow what it reads.
            Instr::Boxed { mem, uses, .. } => {
                if let Some(m) = mem {
                    note(&m.addr, 64);
                }
                for (_, x) in uses {
                    note(x, 64);
                }
            }
            Instr::Opaque { .. } => {}
        }
    }
    // The branch condition, and the exits.
    if let Some(c) = &sb.control {
        note(c, 8);
    }
    for (reg, o) in &sb.exits {
        // What this block's exit expression for `reg` says about the entry value it
        // forwards. A narrowed pass-through costs only the bits it keeps; anything
        // else is a full-width use.
        let carried = b
            .exit_regs
            .iter()
            .find(|(r, _)| r == reg)
            .and_then(|(_, v)| match o {
                Operand::Entry(q) | Operand::Param(q) => preserved_entry_bits(arena, *v, *q),
                _ => None,
            });
        note(o, carried.unwrap_or(64));
    }
    out
}

fn rename_entry_reads(cfg: &Cfg, blocks: &mut [SchedBlock]) {
    let by_id: HashMap<BlockId, &crate::ir::Block> = cfg.blocks.iter().map(|b| (b.id, b)).collect();

    for (sb, b) in blocks.iter_mut().zip(cfg.blocks.iter()) {
        if b.preds.is_empty() {
            continue;
        }

        // Which guest register holds the function-entry value of `q` on arrival, and how many bits of it that register preserves.
        let home_in = |p: &crate::ir::Block, q: Reg| -> Option<(Reg, u32)> {
            let mut found = None;
            for (reg, v) in &p.exit_regs {
                // RSP is never a home. It is the frame anchor rather than a value the allocator places, and a mislocated guest image can show it holding another register's entry value; exactly the case `locate_guest_context` warns about.
                if *reg == Reg::Rsp {
                    continue;
                }
                let Some(bits) = preserved_entry_bits(&cfg.arena, *v, q) else {
                    continue;
                };
                // Identity wins: a value still in its own register needs no rename.
                if *reg == q {
                    return Some((q, bits));
                }
                found = found.or(Some((*reg, bits)));
            }
            found
        };

        // Widest read of each entry value in this block, so a narrowed home is only
        // accepted when every read fits inside what it preserves.
        let read_bits = entry_read_widths(sb, b, &cfg.arena);

        let mut home: HashMap<Reg, Reg> = HashMap::new();
        for q in b.exit_regs.iter().map(|(r, _)| *r) {
            // Guest RSP is the frame anchor, not a value the allocator places, and
            // `frame_extent` reads it directly. Renaming it produced exits wanting a
            // machine RSP that had no live range.
            if q == Reg::Rsp {
                continue;
            }
            let mut agreed: Option<(Reg, u32)> = None;
            let mut ok = true;
            for p in &b.preds {
                let Some(pb) = by_id.get(p) else {
                    ok = false;
                    break;
                };
                match home_in(pb, q) {
                    // Predecessors must agree on the register. Where they disagree on
                    // how much they preserve, the smallest wins, since the rename has
                    // to be correct on every edge.
                    Some((h, bits)) if agreed.is_none_or(|(a, _)| a == h) => {
                        let bits = agreed.map_or(bits, |(_, b)| b.min(bits));
                        agreed = Some((h, bits));
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if let (true, Some((h, bits))) = (ok, agreed) {
                // A narrowed home cannot answer a wider read.
                if read_bits.get(&q).copied().unwrap_or(0) > bits {
                    continue;
                }
                if h != q {
                    home.insert(q, h);
                }
            }
        }
        if home.is_empty() {
            continue;
        }

        let rename = |o: &mut Operand| {
            if let Operand::Entry(q) | Operand::Param(q) = o {
                if let Some(&to) = home.get(q) {
                    *o = Operand::Entry(to);
                }
            }
        };
        for instr in &mut sb.instrs {
            each_operand_mut(instr, &mut |o| rename(o));
        }
        if let Some(c) = &mut sb.control {
            rename(c);
        }
        for (_, o) in &mut sb.exits {
            rename(o);
        }
        if let Some(f) = &mut sb.fused_cmp {
            rename(&mut f.a);
            rename(&mut f.b);
        }
    }
}

fn prune_dead_exits(cfg: &Cfg, blocks: &mut [SchedBlock]) {
    /// Guest registers an operand reads.
    fn reads_of(o: &Operand, into: &mut HashSet<Reg>) {
        if let Operand::Entry(r) | Operand::Param(r) = o {
            into.insert(*r);
        }
    }

    let index: HashMap<BlockId, usize> = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (b.id, i))
        .collect();

    // Registers read by the body: instructions plus the branch condition. These are
    // live-in unconditionally, whatever the exits end up being.
    let body_reads: Vec<HashSet<Reg>> = blocks
        .iter()
        .map(|sb| {
            let mut set = HashSet::new();
            for i in &sb.instrs {
                each_operand(i, &mut |o| reads_of(o, &mut set));
            }
            if let Some(c) = &sb.control {
                reads_of(c, &mut set);
            }
            set
        })
        .collect();

    // A block whose control leaves this function keeps every exit, and any block that
    // can reach one along an edge we cannot reason about must keep them too.
    let opaque_exit: Vec<bool> = cfg
        .blocks
        .iter()
        .map(|b| {
            !matches!(
                b.terminator,
                Terminator::Jump(_)
                    | Terminator::Branch { .. }
                    | Terminator::Switch { .. }
                    | Terminator::Backedge { .. }
                    | Terminator::Adopted { .. }
                    | Terminator::AdoptedSwitch { .. }
            )
        })
        .collect();

    let all_regs: HashSet<Reg> = cfg
        .blocks
        .iter()
        .flat_map(|b| b.exit_regs.iter().map(|(r, _)| *r))
        .collect();

    let mut live_in: Vec<HashSet<Reg>> = body_reads.clone();
    let mut live_out: Vec<HashSet<Reg>> = vec![HashSet::new(); blocks.len()];

    loop {
        let mut changed = false;
        for (i, b) in cfg.blocks.iter().enumerate() {
            // live_out = union of successors' live_in, or everything if the exit is
            // opaque or leads somewhere unrecovered.
            let mut out: HashSet<Reg> = HashSet::new();
            if opaque_exit[i] {
                out = all_regs.clone();
            } else {
                for s in Cfg::successors(b) {
                    match index.get(&s) {
                        Some(&j) => out.extend(live_in[j].iter().copied()),
                        None => out = all_regs.clone(),
                    }
                }
            }

            let returns = matches!(b.terminator, Terminator::Return { .. });
            let mut in_ = body_reads[i].clone();
            for (r, o) in &blocks[i].exits {
                if !out.contains(r) {
                    continue;
                }
                let restored_anyway = returns
                    && !crate::ir::regalloc::VOLATILE.contains(r)
                    && *r != Reg::Rsp
                    && matches!(o, Operand::Entry(e) | Operand::Param(e) if e == r);
                if restored_anyway {
                    continue;
                }
                reads_of(o, &mut in_);
            }
            // A block's params are supplied by its predecessors, so a param the block
            // does not otherwise read still has to arrive.
            for (r, _) in &b.params {
                in_.insert(*r);
            }

            if in_ != live_in[i] || out != live_out[i] {
                live_in[i] = in_;
                live_out[i] = out;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for (i, sb) in blocks.iter_mut().enumerate() {
        // Guest RSP is never dropped: frame recovery reads it to size the frame, and
        // it is not a value the allocator places.
        sb.exits
            .retain(|(r, _)| *r == Reg::Rsp || live_out[i].contains(r));
    }
}

fn schedule_block(cfg: &Cfg, b: &Block) -> SchedBlock {
    let mut s = Scheduler::new(&cfg.arena, b.block_ref, cfg.stack_base);

    // Instruction index and guest RSP of each call, for `retarget_stack_args`.
    let mut call_rsp: Vec<(usize, Option<u64>)> = Vec::new();

    // Effects first, in program order. Scheduling them before the exit registers
    // matters: a load's result is often what a register ends up holding, and
    // emitting the effect first means the register reuses that value rather than
    // scheduling a second, identical load.
    for e in &b.events {
        match e {
            // Only guest-visible effects are emitted. The rest is interpreter
            // bookkeeping and has no place in devirtualized output.
            Event::Store {
                addr,
                value,
                width,
                region,
                ..
            } if region.is_guest() => {
                let addr = s.value(*addr);
                let value = s.value(*value);
                s.instrs.push(Instr::Store {
                    addr,
                    value,
                    disp: 0,
                    width: *width,
                });
            }
            Event::Load {
                addr,
                value,
                width,
                region,
                ..
            } if region.is_guest() => {
                // Scheduling the load's *result* is what places the load, since the
                // value node is itself an `Op::Load`. Going through the value keeps
                // it shared with any later use.
                let _ = s.value(*value);
                let _ = (addr, width);
            }
            Event::Call {
                target,
                site,
                import_slot,
                args,
                rsp,
                ret,
            } => {
                let t = match (target, import_slot) {
                    (_, Some(slot)) => CallTarget::Import { slot: *slot },
                    (Some(t), None) => match cfg.arena.as_const(*t) {
                        Some(a) => CallTarget::Direct(a),
                        None => CallTarget::Indirect(s.value(*t)),
                    },
                    // The callee could not be determined at all. `site` is the
                    // address of the calling instruction inside the VM handler, not
                    // the callee, so using it would fabricate a target.
                    (None, None) => CallTarget::Unknown { site: *site },
                };
                // Schedule each argument expression so the values it needs exist before the call. An argument whose value is a register some earlier call destroyed is undefined, so any register content satisfies it.
                let args: Vec<(Reg, Operand)> = args
                    .iter()
                    .filter(|(_, v)| !matches!(cfg.arena.op(*v), Op::Opaque("clobbered", _)))
                    .map(|(r, v)| (*r, s.value(*v)))
                    .collect();
                // Remember where the guest's RSP was, so the stores feeding this
                // call's stack arguments can be re-anchored to it below.
                call_rsp.push((s.instrs.len(), *rsp));
                s.instrs.push(Instr::Call { target: t, args });
                if let Some(rv) = ret
                    && !s.placed.contains_key(rv)
                {
                    let dst = s.fresh();
                    s.instrs.push(Instr::Opaque {
                        dst,
                        tag: "call_ret",
                        width: Width::W64,
                        // The ABI leaves the return value in RAX.
                        at: Some(Reg::Rax),
                    });
                    s.placed.insert(*rv, Operand::Val(dst));
                }
            }
            Event::Boxed {
                site,
                text,
                bytes,
                mem,
                defs,
                uses,
            } => {
                // The memory operand's address goes through `s.value` like a store's,
                // so it becomes a Frame/Val/Imm operand the allocator can place.
                let mem = mem.as_ref().map(|m| BoxedMemOp {
                    addr: s.value(m.addr),
                    bytes: m.bytes,
                    writes: m.writes,
                });
                // Scheduled before the instruction, so the values exist by the time
                // codegen moves them into the registers the encoding reads.
                let uses: Vec<(Reg, Operand)> = uses
                    .iter()
                    .filter_map(|(r, v)| {
                        let (base, _, _) = crate::ir::lift::decode_reg(*r)?;
                        Some((base, s.value(*v)))
                    })
                    .collect();
                s.instrs.push(Instr::Boxed {
                    site: *site,
                    text: text.clone(),
                    bytes: bytes.clone(),
                    mem,
                    uses,
                });
                // The instruction's register results exist *now*, in the registers the hardware wrote. Placing their opaques here rather than letting `value` place them lazily at the first reader is what keeps the live range anchored to that fact; see `Event::Boxed::defs`.
                for (reg, v) in defs {
                    if s.placed.contains_key(v) {
                        continue;
                    }
                    let Some((at, _, _)) = crate::ir::lift::decode_reg(*reg) else {
                        continue;
                    };
                    let tag = match s.arena.op(*v) {
                        Op::Opaque(t, _) => t,
                        // `defs` is built from opaques this instruction just created,
                        // so anything else means the event was constructed wrongly.
                        _ => continue,
                    };
                    let dst = s.fresh();
                    s.instrs.push(Instr::Opaque {
                        dst,
                        tag,
                        width: Width::W64,
                        at: Some(at),
                    });
                    s.placed.insert(*v, Operand::Val(dst));
                }
            }
            _ => {}
        }
    }

    retarget_stack_args(&mut s.instrs, &call_rsp, s.stack_base);
    reanchor_small_addrs_on_call_ret(&mut s.instrs);

    let control = match &b.terminator {
        Terminator::Branch { predicate, .. } => Some(s.value(*predicate)),
        Terminator::Switch { selector, .. } => Some(s.value(*selector)),
        // A `ret` reads the return address off the stack itself, so the recovered destination needs no register and no instructions. Scheduling it would make it a DCE root and keep alive a load of the return slot whose result nothing can use.
        Terminator::Return { .. } => None,
        _ => None,
    };

    // A register still holding a `call_ret` opaque needs no code: that opaque was minted by `clobber_call` to stand for "whatever the callee left here", and the `call` instruction emitted above already establishes exactly that.
    let mut exits = Vec::new();
    let mut callee_set = Vec::new();
    for (reg, v) in &b.exit_regs {
        if matches!(cfg.arena.op(*v), Op::Opaque("call_ret" | "clobbered", _)) {
            callee_set.push(*reg);
        } else {
            exits.push((*reg, s.value(*v)));
        }
    }

    SchedBlock {
        id: b.id,
        instrs: s.instrs,
        exits,
        callee_set,
        control,
        fused_cmp: None,
        terminator: b.terminator.clone(),
    }
}

fn retarget_stack_args(instrs: &mut [Instr], call_rsp: &[(usize, Option<u64>)], stack_base: u64) {
    for (i, (call_idx, rsp)) in call_rsp.iter().enumerate() {
        // Without a concrete RSP at the call there is nothing to anchor to. Leaving
        // the store entry-relative is wrong, but inventing an offset is worse.
        let Some(rsp) = rsp else { continue };
        // Offset of the call's RSP from entry RSP: the same frame of reference `Operand::Frame` uses.
        let call_off = rsp.wrapping_sub(stack_base) as i64;
        // Scan back only as far as the previous call. A store before that fed the
        // previous call's arguments, not this one's.
        let from = i.checked_sub(1).map_or(0, |p| call_rsp[p].0 + 1);

        // Phase 1: find the stores that look like stack arguments, without committing.
        let mut candidates: Vec<(usize, i64, i64)> = Vec::new();
        for (n, ins) in instrs[from..*call_idx].iter().enumerate() {
            let Instr::Store { addr, disp, .. } = ins else {
                continue;
            };
            let Operand::Frame(off) = *addr else { continue };
            let target = off + *disp;
            if target >= 0 {
                continue;
            }
            // Where the byte actually is, relative to the call's RSP.
            let at = target - call_off;
            // Below the shadow space is the callee's own business, and at or above
            // `SHADOW + 8*MAX_STACK_ARGS` is the guest's frame, not an argument.
            if at < SHADOW_BYTES || at >= SHADOW_BYTES + 8 * MAX_STACK_ARGS {
                continue;
            }
            candidates.push((from + n, target, at));
        }
        if candidates.is_empty() {
            continue;
        }

        let lo = candidates.iter().map(|(_, t, _)| *t).min().unwrap();
        let hi = candidates.iter().map(|(_, t, _)| *t).max().unwrap();
        const SLACK: i64 = 8;
        let aims_into_span = if let Instr::Call { args, .. } = &instrs[*call_idx] {
            args.iter().any(|(_, o)| match o {
                Operand::Frame(f) => *f >= lo - SLACK && *f <= hi + SLACK,
                _ => false,
            })
        } else {
            false
        };
        if aims_into_span {
            continue;
        }

        for (idx, _, at) in candidates {
            let Instr::Store { addr, disp, .. } = &mut instrs[idx] else {
                continue;
            };
            *addr = Operand::OutArg(at as u32);
            *disp = 0;
        }
    }
}

/// Win64 shadow space: the four register arguments' homing slots.
const SHADOW_BYTES: i64 = 0x20;

const MAX_STACK_ARGS: i64 = 12;

/// How many outgoing stack arguments a block's calls need. Counted from the highest `OutArg` byte it writes, since the ABI area is contiguous: a store to `[rsp+0x30]` is argument 7, so arguments 5 and 6 exist too even if this block never writes them.
pub fn stack_arg_count(b: &SchedBlock) -> u32 {
    let mut hi = None;
    for ins in &b.instrs {
        if let Instr::Store {
            addr: Operand::OutArg(n),
            width,
            ..
        } = ins
        {
            let end = *n + width.bytes() as u32;
            hi = Some(hi.map_or(end, |x: u32| x.max(end)));
        }
    }
    // Bytes above the shadow space, rounded up to whole 8-byte slots.
    hi.map_or(0, |h| (h.saturating_sub(SHADOW_BYTES as u32) + 7) / 8)
}

/// Re-anchor a memory access whose address folded to a small constant onto the preceding call's return value. Only accesses below one page are considered: a real absolute access names a global, which lives in the image and is nowhere near zero.
fn reanchor_small_addrs_on_call_ret(instrs: &mut [Instr]) {
    /// Below this, a concrete address cannot be a real global and is taken to be a
    /// lost pointer base.
    const MIN_PLAUSIBLE_ADDR: i64 = 0x1000;

    // Each call's index and the value id of the `call_ret` opaque that follows it.
    // The scheduler emits that opaque immediately after the call, so a single lookahead
    // finds it.
    let mut call_rets: Vec<(usize, Option<ValId>)> = Vec::new();
    for (i, instr) in instrs.iter().enumerate() {
        if !matches!(instr, Instr::Call { .. }) {
            continue;
        }
        let ret = match instrs.get(i + 1) {
            Some(Instr::Opaque {
                dst,
                tag: "call_ret",
                at: Some(Reg::Rax),
                ..
            }) => Some(*dst),
            _ => None,
        };
        call_rets.push((i, ret));
    }

    for (i, instr) in instrs.iter_mut().enumerate() {
        let (addr, disp) = match instr {
            Instr::Store { addr, disp, .. } | Instr::Load { addr, disp, .. } => (addr, disp),
            _ => continue,
        };
        let Operand::Imm(a) = *addr else { continue };
        let target = (a as i64).wrapping_add(*disp);
        if !(0..MIN_PLAUSIBLE_ADDR).contains(&target) {
            continue;
        }
        // The nearest preceding call is the one that produced the pointer.
        let base = call_rets
            .iter()
            .rev()
            .find(|(at, _)| *at < i)
            .and_then(|(_, ret)| *ret);
        if let Some(base) = base {
            *addr = Operand::Val(base);
            *disp = target;
        }
    }
}

/// Visit every operand an instruction reads.
fn each_operand(i: &Instr, f: &mut impl FnMut(&Operand)) {
    match i {
        Instr::Bin { a, b, .. } => {
            f(a);
            f(b);
        }
        Instr::Un { a, .. } | Instr::Cast { a, .. } => f(a),
        Instr::Load { addr, .. } => f(addr),
        Instr::Store { addr, value, .. } => {
            f(addr);
            f(value);
        }
        Instr::Select { cond, a, b, .. } => {
            f(cond);
            f(a);
            f(b);
        }
        Instr::Call { target, args } => {
            if let CallTarget::Indirect(t) = target {
                f(t);
            }
            for (_, a) in args {
                f(a);
            }
        }
        // A boxed instruction reads the address of its memory operand.
        Instr::Boxed { mem, uses, .. } => {
            if let Some(m) = mem {
                f(&m.addr);
            }
            for (_, a) in uses {
                f(a);
            }
        }
        Instr::Opaque { .. } => {}
    }
}

/// As [`each_operand`], for a pass that rewrites operands in place.
fn each_operand_mut(i: &mut Instr, f: &mut impl FnMut(&mut Operand)) {
    match i {
        Instr::Bin { a, b, .. } => {
            f(a);
            f(b);
        }
        Instr::Un { a, .. } | Instr::Cast { a, .. } => f(a),
        Instr::Load { addr, .. } => f(addr),
        Instr::Store { addr, value, .. } => {
            f(addr);
            f(value);
        }
        Instr::Select { cond, a, b, .. } => {
            f(cond);
            f(a);
            f(b);
        }
        Instr::Call { target, args } => {
            if let CallTarget::Indirect(t) = target {
                f(t);
            }
            for (_, a) in args {
                f(a);
            }
        }
        Instr::Boxed { mem, uses, .. } => {
            if let Some(m) = mem {
                f(&mut m.addr);
            }
            // Rewritten like the address, so a `uses` operand is re-anchored by
            // `retarget_stack_args` and `fold_addressing` along with everything else.
            for (_, a) in uses {
                f(a);
            }
        }
        Instr::Opaque { .. } => {}
    }
}

/// The comparison defining `control`, if the branch can absorb it. Requires the value to be defined by exactly one `Eq`/`Ult`/`Slt` and read by nothing except `control`.
fn fusable_cmp(b: &SchedBlock) -> Option<(usize, FusedCmp)> {
    let Some(Operand::Val(cv)) = &b.control else {
        return None;
    };
    let (idx, a, rhs, op, width) = b.instrs.iter().enumerate().find_map(|(i, ins)| {
        let Instr::Bin {
            dst,
            op,
            a,
            b: rhs,
            operand_width,
            ..
        } = ins
        else {
            return None;
        };
        if dst != cv {
            return None;
        }
        matches!(op, BinOp::Eq | BinOp::Ult | BinOp::Slt)
            .then(|| (i, a.clone(), rhs.clone(), *op, *operand_width))
    })?;
    // Any other reader of the comparison keeps it alive.
    let read_elsewhere = |o: &Operand| matches!(o, Operand::Val(v) if v == cv);
    for (j, ins) in b.instrs.iter().enumerate() {
        if j == idx {
            continue;
        }
        let mut hit = false;
        each_operand(ins, &mut |o| hit |= read_elsewhere(o));
        if hit {
            return None;
        }
    }
    if b.exits.iter().any(|(_, o)| read_elsewhere(o)) {
        return None;
    }
    Some((
        idx,
        FusedCmp {
            a,
            b: rhs,
            op,
            width,
        },
    ))
}

/// The lowest and highest frame offsets a block references. The low bound is how much stack the function actually uses, which is what a prologue must reserve.
pub fn frame_extent(b: &SchedBlock) -> Option<(i64, i64)> {
    let mut lo = None;
    let mut hi = None;
    let mut note = |o: &Operand| {
        if let Operand::Frame(off) = o {
            lo = Some(lo.map_or(*off, |x: i64| x.min(*off)));
            hi = Some(hi.map_or(*off, |x: i64| x.max(*off)));
        }
    };
    for i in &b.instrs {
        match i {
            Instr::Bin { a, b: rhs, .. } => {
                note(a);
                note(rhs);
            }
            Instr::Un { a, .. } | Instr::Cast { a, .. } | Instr::Load { addr: a, .. } => note(a),
            Instr::Select {
                cond, a, b: rhs, ..
            } => {
                note(cond);
                note(a);
                note(rhs);
            }
            Instr::Store { addr, value, .. } => {
                note(addr);
                note(value);
            }
            Instr::Call { target, args } => {
                if let CallTarget::Indirect(t) = target {
                    note(t);
                }
                for (_, a) in args {
                    note(a);
                }
            }
            _ => {}
        }
    }
    for (_, o) in &b.exits {
        note(o);
    }
    if let Some(c) = &b.control {
        note(c);
    }
    lo.zip(hi)
}

pub fn fold_addressing(b: &mut SchedBlock) -> usize {
    // Base and offset of every `Add reg, const`, by destination.
    let mut adds: HashMap<ValId, (Operand, i64)> = HashMap::new();
    for instr in &b.instrs {
        if let Instr::Bin {
            dst,
            op: BinOp::Add,
            a,
            b: Operand::Imm(c),
            width: Width::W64,
            ..
        } = instr
        {
            // A `Frame` base already encodes an RSP displacement that the frame
            // layout owns, so folding into it would double-count.
            if !matches!(a, Operand::Imm(_) | Operand::Frame(_)) {
                adds.insert(*dst, (*a, *c as i64));
            }
        }
    }

    let mut folded = 0usize;
    for instr in &mut b.instrs {
        let (addr, disp) = match instr {
            Instr::Load { addr, disp, .. } | Instr::Store { addr, disp, .. } => (addr, disp),
            _ => continue,
        };
        let Operand::Val(v) = *addr else { continue };
        let Some((base, off)) = adds.get(&v) else {
            continue;
        };
        let Some(total) = disp.checked_add(*off) else {
            continue;
        };
        if total < i32::MIN as i64 || total > i32::MAX as i64 {
            continue;
        }
        *addr = *base;
        *disp = total;
        folded += 1;
    }
    folded
}

fn rewrite_masked_sign_tests(b: &mut SchedBlock) -> usize {
    // A value is safe to bypass only if the comparison is its sole reader; otherwise
    // the `and` has to stay and rewriting the comparison would leave both.
    let mut readers: HashMap<ValId, usize> = HashMap::new();
    for ins in &b.instrs {
        each_operand(ins, &mut |o| {
            if let Operand::Val(v) = o {
                *readers.entry(*v).or_default() += 1;
            }
        });
    }
    for (_, o) in &b.exits {
        if let Operand::Val(v) = o {
            *readers.entry(*v).or_default() += 1;
        }
    }
    if let Some(Operand::Val(v)) = &b.control {
        *readers.entry(*v).or_default() += 1;
    }

    // What each value is defined as, so a comparison can look up its operand.
    let defs: HashMap<ValId, (BinOp, Operand, Operand, Width)> = b
        .instrs
        .iter()
        .filter_map(|ins| match ins {
            Instr::Bin {
                dst,
                op,
                a,
                b: rhs,
                operand_width,
                ..
            } => Some((*dst, (*op, a.clone(), rhs.clone(), *operand_width))),
            _ => None,
        })
        .collect();

    let mut rewritten = 0;
    for i in 0..b.instrs.len() {
        let Instr::Bin {
            op: BinOp::Eq,
            a,
            b: rhs,
            operand_width,
            ..
        } = &b.instrs[i]
        else {
            continue;
        };
        let width = *operand_width;
        let sign = 1u64 << (width.bits() - 1);
        // `x == C` or `C == x`; equality is symmetric so accept either.
        let masked = match (a, rhs) {
            (Operand::Val(v), Operand::Imm(c)) if *c == sign => *v,
            (Operand::Imm(c), Operand::Val(v)) if *c == sign => *v,
            _ => continue,
        };
        if readers.get(&masked).copied().unwrap_or(0) != 1 {
            continue;
        }
        // The masked value must be `x & SIGN` at the same width, with the same bit.
        let Some((BinOp::And, and_a, and_b, and_w)) = defs.get(&masked) else {
            continue;
        };
        if *and_w != width {
            continue;
        }
        let x = match (and_a, and_b) {
            (x, Operand::Imm(c)) if *c == sign => x.clone(),
            (Operand::Imm(c), x) if *c == sign => x.clone(),
            _ => continue,
        };
        let Instr::Bin { op, a, b: rhs, .. } = &mut b.instrs[i] else {
            unreachable!()
        };
        *op = BinOp::Slt;
        *a = x;
        *rhs = Operand::Imm(0);
        rewritten += 1;
    }
    rewritten
}

/// Frame slots whose address a callee could hold, so their stores must survive a call. DSE's problem at a call is that it cannot see the callee's reads.
#[derive(Debug, Default, Clone)]
pub struct FrameEscapes {
    /// Half-open byte ranges of frame offsets a callee may access.
    runs: Vec<(i64, i64)>,
    /// Set when an escaping frame address could not be attributed to a run, which
    /// forces the conservative blanket behaviour.
    unattributed: bool,
}

impl FrameEscapes {
    /// Every frame slot may be read. The behaviour before escape analysis existed.
    #[cfg(test)]
    pub fn conservative() -> Self {
        Self {
            runs: Vec::new(),
            unattributed: true,
        }
    }

    /// Whether a callee may observe the frame slot at `key`.
    pub fn may_escape(&self, key: i64) -> bool {
        self.unattributed || self.runs.iter().any(|&(lo, hi)| key >= lo && key < hi)
    }
}

/// Compute [`FrameEscapes`] over every block of a function.
pub fn analyze_frame_escapes(blocks: &[SchedBlock]) -> FrameEscapes {
    // Every (offset, length) a frame slot is directly accessed at, and every offset
    // that leaks as a value.
    let mut accesses: Vec<(i64, i64)> = Vec::new();
    let mut escaped: Vec<i64> = Vec::new();

    for b in blocks {
        // A `Frame` reaching here is an address used as a value, not as an address.
        let mut leak = |o: &Operand| {
            if let Operand::Frame(off) = o {
                escaped.push(*off);
            }
        };
        for ins in &b.instrs {
            match ins {
                Instr::Store {
                    addr,
                    value,
                    disp,
                    width,
                } => {
                    if let Operand::Frame(off) = addr {
                        accesses.push((off + disp, width.bytes() as i64));
                    } else {
                        leak(addr);
                    }
                    leak(value);
                }
                Instr::Load {
                    addr, disp, width, ..
                } => {
                    if let Operand::Frame(off) = addr {
                        accesses.push((off + disp, width.bytes() as i64));
                    } else {
                        leak(addr);
                    }
                }
                // A boxed instruction's memory operand is an access of a width this
                // IR does not carry. Take the widest an x86 memory operand can have,
                // so the run covering it is never under-sized.
                Instr::Boxed { mem, uses, .. } => {
                    if let Some(m) = mem {
                        if let Operand::Frame(off) = &m.addr {
                            accesses.push((*off, 64));
                        } else {
                            leak(&m.addr);
                        }
                    }
                    // A frame address handed to an implicit register read escapes just
                    // as much as one in a memory operand: `rep stosb` takes its
                    // destination in RDI, so a frame slot reaching RDI is written
                    // through by an instruction this IR does not model.
                    for (_, a) in uses {
                        leak(a);
                    }
                }
                Instr::Call { target, args } => {
                    if let CallTarget::Indirect(v) = target {
                        leak(v);
                    }
                    for (_, a) in args {
                        leak(a);
                    }
                }
                Instr::Bin { a, b: rhs, .. } => {
                    leak(a);
                    leak(rhs);
                }
                Instr::Un { a, .. } | Instr::Cast { a, .. } => leak(a),
                Instr::Select {
                    cond, a, b: rhs, ..
                } => {
                    leak(cond);
                    leak(a);
                    leak(rhs);
                }
                Instr::Opaque { .. } => {}
            }
        }
        // A frame address that leaves the block as a register value escapes: another
        // block, or the caller, can do anything with it.
        for (_, o) in &b.exits {
            leak(o);
        }
        if let Some(o) = &b.control {
            leak(o);
        }
        if let Some(fc) = &b.fused_cmp {
            leak(&fc.a);
            leak(&fc.b);
        }
    }

    // Merge accesses into maximal runs. Sort by start, then extend while the next
    // access begins at or before the current run's end: touching accesses are one
    // object, which is exactly the shape of a byte-at-a-time string build.
    accesses.sort_unstable();
    let mut runs: Vec<(i64, i64)> = Vec::new();
    for (off, len) in accesses {
        let end = off + len.max(1);
        match runs.last_mut() {
            Some(last) if off <= last.1 => last.1 = last.1.max(end),
            _ => runs.push((off, end)),
        }
    }

    // Bound every escape to a region. An escape inside a run taints that run.
    let mut out = FrameEscapes::default();
    if runs.is_empty() {
        out.unattributed = !escaped.is_empty();
    }
    for off in escaped {
        let region = match runs.iter().find(|&&(lo, hi)| off >= lo && off < hi) {
            Some(&r) => r,
            None => {
                // The hole containing `off`: from the end of the nearest run at or
                // below it to the start of the nearest run above it. Open-ended when
                // the escape is beyond the outermost run.
                let lo = runs
                    .iter()
                    .map(|&(_, hi)| hi)
                    .filter(|&hi| hi <= off)
                    .max()
                    .unwrap_or(i64::MIN);
                let hi = runs
                    .iter()
                    .map(|&(lo, _)| lo)
                    .filter(|&lo| lo > off)
                    .min()
                    .unwrap_or(i64::MAX);
                (lo, hi)
            }
        };
        if !out.runs.contains(&region) {
            out.runs.push(region);
        }
    }

    // Non-negative offsets are the caller's: the return address at `+0` and the
    // shadow space above it. Those are readable by the caller and by anything it
    // hands the frame pointer to, whatever this function does with their addresses.
    out.runs.push((0, i64::MAX));
    out
}

/// Frame byte ranges that some block reads, or that a callee could.
///
/// Reads are collected as runs, the same way [`analyze_frame_escapes`] collects
/// accesses, because a wide read of a slot written narrowly still observes it.
#[derive(Debug, Default, Clone)]
pub struct FrameReads {
    /// Half-open byte ranges any block loads from.
    runs: Vec<(i64, i64)>,
}

impl FrameReads {
    /// Whether any block reads a byte in `[key, key+len)`.
    fn overlaps(&self, key: i64, len: i64) -> bool {
        let end = key + len.max(1);
        self.runs.iter().any(|&(lo, hi)| key < hi && lo < end)
    }
}

/// Collect every frame range a block loads from, across the whole function. Only `Load` and `Boxed` memory operands count.
pub fn analyze_frame_reads(blocks: &[SchedBlock]) -> FrameReads {
    let mut reads: Vec<(i64, i64)> = Vec::new();
    for b in blocks {
        for ins in &b.instrs {
            match ins {
                Instr::Load {
                    addr: Operand::Frame(off),
                    disp,
                    width,
                    ..
                } => {
                    reads.push((off + disp, width.bytes() as i64));
                }
                // A boxed instruction's memory operand is assumed readable. `writes` is not the complement of "reads": it only records that operand 0 is memory, which is equally true of `add [mem], 1`, a read-modify-write.
                Instr::Boxed { mem: Some(m), .. } => {
                    if let Operand::Frame(off) = &m.addr {
                        reads.push((*off, m.bytes.max(1) as i64));
                    }
                }
                _ => {}
            }
        }
    }
    reads.sort_unstable();
    let mut runs: Vec<(i64, i64)> = Vec::new();
    for (off, len) in reads {
        let end = off + len.max(1);
        match runs.last_mut() {
            Some(last) if off <= last.1 => last.1 = last.1.max(end),
            _ => runs.push((off, end)),
        }
    }
    FrameReads { runs }
}

/// Delete stores to frame slots that no block reads and no callee can reach. Soundness rests on both analyses together. Returns the number of stores removed.
pub fn drop_write_only_frame_slots(
    blocks: &mut [SchedBlock],
    reads: &FrameReads,
    escapes: &FrameEscapes,
) -> usize {
    let mut removed = 0;
    for b in blocks {
        b.instrs.retain(|ins| {
            let Instr::Store {
                addr: Operand::Frame(off),
                disp,
                width,
                ..
            } = ins
            else {
                return true;
            };
            let key = off + disp;
            let len = width.bytes() as i64;
            if reads.overlaps(key, len) || escapes.may_escape(key) {
                return true;
            }
            removed += 1;
            false
        });
    }
    removed
}

/// Iterated to a fixed point, since removing one instruction can orphan another.
///
/// Only reachable from tests. The pipeline calls [`eliminate_dead_with`] directly,
/// because it has real escape information and this would discard it.
#[cfg(test)]
pub fn eliminate_dead(b: &mut SchedBlock) -> usize {
    eliminate_dead_with(b, &FrameEscapes::conservative())
}

/// [`eliminate_dead`], told which frame slots a callee may read.
pub fn eliminate_dead_with(b: &mut SchedBlock, escapes: &FrameEscapes) -> usize {
    // Turn masked sign tests into signed comparisons first: it changes which value
    // the comparison reads, so the `and` it bypasses only becomes collectable once
    // this has run.
    rewrite_masked_sign_tests(b);
    // Liveness by a single backward walk. Exact in one pass for straight-line code: every use of a value comes after its definition, so by the time the definition is reached all of its uses have been seen.
    let fused = fusable_cmp(b);
    if let Some((_, fc)) = &fused {
        b.fused_cmp = Some(fc.clone());
    }

    let mut used: HashSet<ValId> = HashSet::new();
    for (_, o) in &b.exits {
        if let Operand::Val(v) = o {
            used.insert(*v);
        }
    }
    // A fused comparison is tested from its operands, so `control` no longer reads the boolean.
    match &fused {
        Some((_, fc)) => {
            for o in [&fc.a, &fc.b] {
                if let Operand::Val(v) = o {
                    used.insert(*v);
                }
            }
        }
        None => {
            if let Some(Operand::Val(v)) = &b.control {
                used.insert(*v);
            }
        }
    }

    // Track which memory addresses are known to be overwritten: a store to `[base+k]` that is followed only by another store to `[base+k]` before any read is dead.
    let mut overwritten: HashSet<(ValId, i64)> = HashSet::new();
    let mut overwritten_frame: HashSet<i64> = HashSet::new();

    let mut keep = vec![false; b.instrs.len()];
    for (i, instr) in b.instrs.iter().enumerate().rev() {
        let live = match instr {
            // A store to an address known to be overwritten is dead. Calls and
            // boxed instructions are always live.
            Instr::Store { addr, disp, .. } => {
                match addr {
                    Operand::Val(v) => {
                        let slot = (*v, *disp);
                        if overwritten.contains(&slot) {
                            false
                        } else {
                            overwritten.insert(slot);
                            true
                        }
                    }
                    // A frame-relative store is local to this activation record.
                    // Track it the same way: if the slot is overwritten before the
                    // next read, the earlier store is dead.
                    Operand::Frame(off) => {
                        let key = off + disp;
                        if overwritten_frame.contains(&key) {
                            false
                        } else {
                            overwritten_frame.insert(key);
                            true
                        }
                    }
                    // Stores through absolute constants are true globals; visible
                    // to other threads and to any code this block does not model.
                    _ => true,
                }
            }
            Instr::Call { .. } | Instr::Boxed { .. } => {
                overwritten.clear();
                overwritten_frame.retain(|key| !escapes.may_escape(*key));
                true
            }
            Instr::Load {
                dst, addr, disp, ..
            } => {
                // A load from a slot makes subsequent stores to that slot live: we
                // cannot prove the store is dead without knowing the load's result
                // is unused, which requires walking forward to find its uses.
                // Conservatively clearing the slot here keeps those stores.
                if let Operand::Val(v) = addr {
                    overwritten.remove(&(*v, *disp));
                }
                if let Operand::Frame(off) = addr {
                    overwritten_frame.remove(&(off + disp));
                }
                used.contains(dst)
            }
            Instr::Bin { dst, .. }
            | Instr::Un { dst, .. }
            | Instr::Cast { dst, .. }
            | Instr::Select { dst, .. }
            | Instr::Opaque { dst, .. } => used.contains(dst),
        };
        keep[i] = live;
        if !live {
            continue;
        }
        let mut note = |o: &Operand| {
            if let Operand::Val(v) = o {
                used.insert(*v);
            }
        };
        match instr {
            Instr::Store { addr, value, .. } => {
                note(addr);
                note(value);
            }
            Instr::Call { target, args } => {
                if let CallTarget::Indirect(v) = target {
                    note(v);
                }
                // Argument values are used by the call, so they must stay live.
                for (_, a) in args {
                    note(a);
                }
            }
            Instr::Bin { a, b: rhs, .. } => {
                note(a);
                note(rhs);
            }
            Instr::Un { a, .. } | Instr::Cast { a, .. } => note(a),
            Instr::Load { addr, .. } => note(addr),
            Instr::Select {
                cond, a, b: rhs, ..
            } => {
                note(cond);
                note(a);
                note(rhs);
            }
            // A boxed instruction reads the address of its memory operand.
            Instr::Boxed { mem, uses, .. } => {
                if let Some(m) = mem {
                    note(&m.addr);
                }
                for (_, a) in uses {
                    note(a);
                }
            }
            Instr::Opaque { .. } => {}
        }
    }

    let before = b.instrs.len();
    let mut n = 0usize;
    b.instrs.retain(|_| {
        let k = keep[n];
        n += 1;
        k
    });
    before - b.instrs.len()
}

/// Render a call's argument list, e.g. `(rcx=v3, rdx=0x0)`.
fn render_args(args: &[(Reg, Operand)]) -> String {
    if args.is_empty() {
        return String::new();
    }
    let parts: Vec<String> = args
        .iter()
        .map(|(r, v)| format!("{}={}", r.name(), render_operand(*v)))
        .collect();
    format!("({})", parts.join(", "))
}

/// Render a scheduled block, naming import calls from `imports`.
/// Schedule a single block without running DCE, so a test can measure its effect.
#[cfg(test)]
pub fn schedule_one(cfg: &Cfg, b: &Block) -> SchedBlock {
    schedule_block(cfg, b)
}

pub fn render_block_with(b: &SchedBlock, imports: &HashMap<u64, String>) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "{}:", b.id);
    for i in &b.instrs {
        let text = match i {
            Instr::Call {
                target: CallTarget::Import { slot },
                args,
            } => imports
                .get(slot)
                .map(|n| format!("call {n}{}", render_args(args)))
                .unwrap_or_else(|| render_instr(i)),
            _ => render_instr(i),
        };
        let _ = writeln!(out, "    {text}");
    }
    // When a comparison is fused into the branch, `control` still names the
    // removed boolean, which makes the dump look broken. Show the fused condition
    // instead: that's what codegen uses, and it exists in the instruction stream.
    if let Some(fc) = &b.fused_cmp {
        let _ = writeln!(
            out,
            "    control = {} {} {}",
            render_operand(fc.a),
            match fc.op {
                crate::ir::expr::BinOp::Eq => "==",
                crate::ir::expr::BinOp::Ult => "<",
                crate::ir::expr::BinOp::Slt => "s<",
                _ => "?",
            },
            render_operand(fc.b)
        );
    } else if let Some(c) = b.control {
        let _ = writeln!(out, "    control = {}", render_operand(c));
    }
    let live: Vec<String> = b
        .exits
        .iter()
        .filter(|(reg, o)| !matches!(o, Operand::Entry(e) | Operand::Param(e) if e == reg))
        .map(|(reg, o)| format!("{}={}", reg.name(), render_operand(*o)))
        .collect();
    if !live.is_empty() {
        let _ = writeln!(out, "    exits: {}", live.join(", "));
    }
    if !b.callee_set.is_empty() {
        let names: Vec<&str> = b.callee_set.iter().map(|r| r.name()).collect();
        let _ = writeln!(out, "    (callee-set: {})", names.join(", "));
    }
    out
}

fn render_operand(o: Operand) -> String {
    match o {
        Operand::Val(v) => v.to_string(),
        Operand::Imm(c) => format!("{c:#x}"),
        Operand::Param(r) => format!("in.{}", r.name()),
        Operand::Entry(r) => r.name().to_string(),
        Operand::Frame(o) if o < 0 => format!("rsp-{:#x}", -o),
        Operand::Frame(o) => format!("rsp+{o:#x}"),
        // Distinguished from a `Frame` in the dump because it is anchored to the
        // call's RSP, not entry RSP, and reading it as a frame offset would be wrong.
        Operand::OutArg(n) => format!("arg[rsp+{n:#x}]"),
    }
}

/// Render a memory operand with its folded displacement.
fn mem(base: String, disp: i64) -> String {
    if disp == 0 {
        base
    } else if disp < 0 {
        format!("{base}-{:#x}", -disp)
    } else {
        format!("{base}+{disp:#x}")
    }
}

fn render_instr(i: &Instr) -> String {
    let o = render_operand;
    match i {
        Instr::Bin {
            dst,
            op,
            a,
            b,
            width,
            ..
        } => {
            format!("{dst}:{} = {:?} {}, {}", width.bits(), op, o(*a), o(*b))
        }
        Instr::Un { dst, op, a, width } => {
            format!("{dst}:{} = {:?} {}", width.bits(), op, o(*a))
        }
        Instr::Cast {
            dst,
            kind,
            a,
            from,
            to,
        } => {
            format!("{dst}:{} = {:?}{} {}", to.bits(), kind, from.bits(), o(*a))
        }
        Instr::Select {
            dst,
            cond,
            a,
            b,
            width,
        } => {
            format!(
                "{dst}:{} = select {}, {}, {}",
                width.bits(),
                o(*cond),
                o(*a),
                o(*b)
            )
        }
        Instr::Load {
            dst,
            addr,
            disp,
            width,
        } => {
            format!("{dst}:{} = load [{}]", width.bits(), mem(o(*addr), *disp))
        }
        Instr::Store {
            addr,
            value,
            disp,
            width,
        } => {
            format!(
                "store{} [{}], {}",
                width.bits(),
                mem(o(*addr), *disp),
                o(*value)
            )
        }
        Instr::Call { target, args } => {
            let t = match target {
                CallTarget::Direct(a) => format!("call {a:#x}"),
                CallTarget::Indirect(v) => format!("call [{}]", o(*v)),
                CallTarget::Import { slot } => format!("call import [{slot:#x}]"),
                CallTarget::Unknown { site } => format!("call <unrecovered> @{site:#x}"),
            };
            format!("{t}{}", render_args(args))
        }
        Instr::Boxed {
            site,
            text,
            mem,
            uses,
            ..
        } => {
            // Show implicit register reads that are not visible in the encoding.
            let u = if uses.is_empty() {
                String::new()
            } else {
                let parts: Vec<String> = uses
                    .iter()
                    .map(|(r, o)| format!("{r:?}={}", render_operand(o.clone())).to_lowercase())
                    .collect();
                format!("({})", parts.join(", "))
            };
            match mem {
                Some(m) => format!(
                    "boxed @{site:#x} {text}{u}   ; mem {} {} bytes {}",
                    render_operand(m.addr.clone()),
                    m.bytes,
                    if m.writes { "write" } else { "read" }
                ),
                None => format!("boxed @{site:#x} {text}{u}"),
            }
        }
        Instr::Opaque {
            dst,
            tag,
            width,
            at,
        } => match at {
            Some(r) => format!("{dst}:{} = opaque {tag} @{r:?}", width.bits()),
            None => format!("{dst}:{} = opaque {tag}", width.bits()),
        },
    }
}

/// Total instructions across a scheduled function, for sizing.
pub fn instr_count(blocks: &[SchedBlock]) -> usize {
    blocks.iter().map(|b| b.instrs.len()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::expr::{Arena, BlockRef};
    use crate::ir::lift::Region;
    use crate::ir::{Block, BlockId, Cfg, Terminator};
    use crate::vm::DEFAULT_STACK_BASE;

    const STACK_BASE: u64 = DEFAULT_STACK_BASE;

    /// An immediate store to a frame slot, for the narrowing tests.
    fn fstore(a: i64, v: u64, w: Width) -> Instr {
        Instr::Store {
            addr: Operand::Frame(a),
            disp: 0,
            value: Operand::Imm(v),
            width: w,
        }
    }

    fn bid(h: u64) -> BlockId {
        BlockId {
            handler: h,
            vip: None,
        }
    }

    fn empty_sb(id: BlockId, exits: Vec<(Reg, Operand)>, t: Terminator) -> SchedBlock {
        SchedBlock {
            id,
            instrs: Vec::new(),
            exits,
            callee_set: Vec::new(),
            control: None,
            terminator: t,
            fused_cmp: None,
        }
    }

    /// A masked sign test becomes a signed comparison against zero, and the unused
    /// mask operation is removed by DCE.
    #[test]
    fn a_masked_sign_test_becomes_a_signed_comparison() {
        for (width, sign) in [
            (Width::W32, 0x8000_0000u64),
            (Width::W8, 0x80),
            (Width::W64, 0x8000_0000_0000_0000),
            (Width::W16, 0x8000),
        ] {
            let x = ValId(2);
            let and = ValId(1);
            let cmp = ValId(0);
            let mut b = empty_sb(
                bid(0x1000),
                vec![(Reg::Rax, Operand::Val(cmp))],
                Terminator::AdoptedReturn,
            );
            b.instrs.push(Instr::Bin {
                dst: and,
                op: BinOp::And,
                a: Operand::Val(x),
                b: Operand::Imm(sign),
                width,
                operand_width: width,
            });
            b.instrs.push(Instr::Bin {
                dst: cmp,
                op: BinOp::Eq,
                a: Operand::Val(and),
                b: Operand::Imm(sign),
                width: Width::W8,
                operand_width: width,
            });
            eliminate_dead(&mut b);
            let bins: Vec<_> = b
                .instrs
                .iter()
                .filter_map(|i| match i {
                    Instr::Bin { op, a, b: rhs, .. } => Some((*op, a.clone(), rhs.clone())),
                    _ => None,
                })
                .collect();
            assert_eq!(
                bins,
                vec![(BinOp::Slt, Operand::Val(x), Operand::Imm(0))],
                "{width:?}: (x & {sign:#x}) == {sign:#x} is x < 0, and the mask must go"
            );
        }
    }

    #[test]
    fn a_masked_sign_test_is_only_rewritten_when_it_is_sound() {
        // Not the sign bit.
        let mut b = empty_sb(
            bid(0x1000),
            vec![(Reg::Rax, Operand::Val(ValId(0)))],
            Terminator::AdoptedReturn,
        );
        b.instrs.push(Instr::Bin {
            dst: ValId(1),
            op: BinOp::And,
            a: Operand::Val(ValId(2)),
            b: Operand::Imm(0x10),
            width: Width::W32,
            operand_width: Width::W32,
        });
        b.instrs.push(Instr::Bin {
            dst: ValId(0),
            op: BinOp::Eq,
            a: Operand::Val(ValId(1)),
            b: Operand::Imm(0x10),
            width: Width::W8,
            operand_width: Width::W32,
        });
        assert_eq!(
            rewrite_masked_sign_tests(&mut b),
            0,
            "bit 4 is not the sign bit"
        );

        // Sign bit, but the mask result is read twice.
        let mut b = empty_sb(
            bid(0x1000),
            vec![
                (Reg::Rax, Operand::Val(ValId(0))),
                (Reg::Rbx, Operand::Val(ValId(1))),
            ],
            Terminator::AdoptedReturn,
        );
        b.instrs.push(Instr::Bin {
            dst: ValId(1),
            op: BinOp::And,
            a: Operand::Val(ValId(2)),
            b: Operand::Imm(0x8000_0000),
            width: Width::W32,
            operand_width: Width::W32,
        });
        b.instrs.push(Instr::Bin {
            dst: ValId(0),
            op: BinOp::Eq,
            a: Operand::Val(ValId(1)),
            b: Operand::Imm(0x8000_0000),
            width: Width::W8,
            operand_width: Width::W32,
        });
        assert_eq!(
            rewrite_masked_sign_tests(&mut b),
            0,
            "the masked value escapes, so the `and` cannot be bypassed"
        );
    }

    #[test]
    fn a_private_frame_slot_is_dead_across_a_call() {
        let mut b = empty_sb(bid(0x1000), Vec::new(), Terminator::AdoptedReturn);
        b.instrs.push(Instr::Store {
            addr: Operand::Frame(-0x68),
            value: Operand::Imm(1),
            disp: 0,
            width: Width::W64,
        });
        b.instrs.push(Instr::Call {
            target: CallTarget::Direct(0x140001000),
            args: Vec::new(),
        });
        b.instrs.push(Instr::Store {
            addr: Operand::Frame(-0x68),
            value: Operand::Imm(2),
            disp: 0,
            width: Width::W64,
        });
        let escapes = analyze_frame_escapes(std::slice::from_ref(&b));
        assert!(
            !escapes.may_escape(-0x68),
            "nothing takes the slot's address"
        );
        eliminate_dead_with(&mut b, &escapes);
        let stores = b
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::Store { .. }))
            .count();
        assert_eq!(
            stores, 1,
            "the overwritten store must go; got {:?}",
            b.instrs
        );
    }

    /// The same slot survives once its address escapes.
    ///
    /// The guard that makes the pass safe. A callee handed a pointer into the frame can
    /// read the slot, so the earlier store is observable even though this block has no
    /// load of it.
    #[test]
    fn an_address_taken_frame_slot_survives_a_call() {
        let mut b = empty_sb(bid(0x1000), Vec::new(), Terminator::AdoptedReturn);
        b.instrs.push(Instr::Store {
            addr: Operand::Frame(-0x68),
            value: Operand::Imm(1),
            disp: 0,
            width: Width::W64,
        });
        // The address of the slot is handed to the callee.
        b.instrs.push(Instr::Call {
            target: CallTarget::Direct(0x140001000),
            args: vec![(Reg::Rcx, Operand::Frame(-0x68))],
        });
        b.instrs.push(Instr::Store {
            addr: Operand::Frame(-0x68),
            value: Operand::Imm(2),
            disp: 0,
            width: Width::W64,
        });
        let escapes = analyze_frame_escapes(std::slice::from_ref(&b));
        assert!(
            escapes.may_escape(-0x68),
            "the address was passed to the callee"
        );
        eliminate_dead_with(&mut b, &escapes);
        let stores = b
            .instrs
            .iter()
            .filter(|i| matches!(i, Instr::Store { .. }))
            .count();
        assert_eq!(stores, 2, "both stores must survive; got {:?}", b.instrs);
    }

    /// An escape into a buffer taints the whole buffer, not just the byte named. A string built a byte at a time is one object, and `&buf[0]` licenses the callee to read all of it.
    #[test]
    fn an_escape_into_a_buffer_covers_its_interior() {
        let mut b = empty_sb(bid(0x1000), Vec::new(), Terminator::AdoptedReturn);
        for i in 0..6i64 {
            b.instrs.push(Instr::Store {
                addr: Operand::Frame(-0x37 + i),
                value: Operand::Imm(0x41),
                disp: 0,
                width: Width::W8,
            });
        }
        // Only the base is handed over.
        b.instrs.push(Instr::Call {
            target: CallTarget::Direct(0x140001000),
            args: vec![(Reg::Rcx, Operand::Frame(-0x37))],
        });
        let escapes = analyze_frame_escapes(std::slice::from_ref(&b));
        for i in 0..6i64 {
            assert!(
                escapes.may_escape(-0x37 + i),
                "byte {i} of the escaped buffer must be treated as readable"
            );
        }
        // A slot outside the buffer is still private.
        assert!(!escapes.may_escape(-0x68));
    }
    #[test]
    fn a_store_into_the_outgoing_argument_area_is_anchored_to_the_call() {
        let stack_base = STACK_BASE;
        let call_rsp = stack_base - 0x58;
        let mut instrs = vec![
            Instr::Store {
                addr: Operand::Frame(-0x38),
                value: Operand::Imm(0),
                disp: 0,
                width: Width::W32,
            },
            Instr::Call {
                target: CallTarget::Direct(0x140001000),
                args: Vec::new(),
            },
        ];
        retarget_stack_args(&mut instrs, &[(1, Some(call_rsp))], stack_base);
        assert!(
            matches!(
                instrs[0],
                Instr::Store {
                    addr: Operand::OutArg(0x20),
                    disp: 0,
                    ..
                }
            ),
            "entry-0x38 with RSP at entry-0x58 is argument 5, at call_rsp+0x20; got {:?}",
            instrs[0]
        );
    }

    #[test]
    fn a_store_to_the_callers_frame_stays_entry_relative() {
        let stack_base = STACK_BASE;
        let call_rsp = stack_base - 0x58;
        let mut instrs = vec![
            Instr::Store {
                addr: Operand::Frame(8),
                value: Operand::Entry(Reg::Rbx),
                disp: 0,
                width: Width::W64,
            },
            Instr::Call {
                target: CallTarget::Direct(0x140001000),
                args: Vec::new(),
            },
        ];
        retarget_stack_args(&mut instrs, &[(1, Some(call_rsp))], stack_base);
        assert!(
            matches!(
                instrs[0],
                Instr::Store {
                    addr: Operand::Frame(8),
                    ..
                }
            ),
            "entry+8 is the caller's frame, not an argument; got {:?}",
            instrs[0]
        );
    }

    #[test]
    fn stack_arguments_survive_a_later_store_to_the_same_slot() {
        // v0 is the stack base. Store to [v0+0x20], call, then store the same slot
        // again. Only the *second* store may be considered redundant-free; the first
        // is read by the call and must be kept.
        let mut arena = Arena::new();
        let dest = arena.constant(0, Width::W64);
        let mut sb = empty_sb(bid(0x1000), Vec::new(), Terminator::Return { dest });
        let slot = |v| Instr::Store {
            addr: Operand::Val(ValId(0)),
            value: Operand::Imm(v),
            disp: 0x20,
            width: Width::W64,
        };
        sb.instrs = vec![
            slot(0xaa),
            Instr::Call {
                target: CallTarget::Direct(0x140001000),
                args: Vec::new(),
            },
            slot(0xbb),
        ];
        eliminate_dead(&mut sb);

        let stores: Vec<u64> = sb
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::Store {
                    value: Operand::Imm(v),
                    ..
                } => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(
            stores,
            vec![0xaa, 0xbb],
            "the store before the call is an argument the callee reads and must not be              eliminated by the store after it"
        );
    }

    /// A wide store whose upper bytes are overwritten narrows to the surviving prefix.
    #[test]
    fn the_address_of_a_boxed_instruction_keeps_its_definition_alive() {
        // The address arithmetic for a boxed memory operand is a real use. When `each_operand` skipped `Instr::Boxed`, this `Add` looked dead and was deleted, and codegen then aimed the boxed instruction at whatever the scratch register happened to hold.
        let base = ValId(0);
        let addr = ValId(1);
        let mut b = empty_sb(bid(0x1000), vec![], Terminator::AdoptedReturn);
        b.instrs.push(Instr::Opaque {
            dst: base,
            tag: "call_ret",
            width: Width::W64,
            at: Some(Reg::Rax),
        });
        b.instrs.push(Instr::Bin {
            dst: addr,
            op: BinOp::Add,
            a: Operand::Val(base),
            b: Operand::Imm(0x10),
            width: Width::W64,
            operand_width: Width::W64,
        });
        b.instrs.push(Instr::Boxed {
            site: 0x1400b0adb,
            text: "movups xmm1,[rcx+10h]".into(),
            bytes: vec![0x0F, 0x10, 0x49, 0x10],
            mem: Some(BoxedMemOp {
                addr: Operand::Val(addr),
                bytes: 16,
                writes: false,
            }),
            uses: Vec::new(),
        });

        eliminate_dead(&mut b);

        assert!(
            b.instrs
                .iter()
                .any(|i| matches!(i, Instr::Bin { dst, .. } if *dst == addr)),
            "the Add computing the boxed operand's address was deleted as dead"
        );
        assert!(
            b.instrs.iter().any(|i| matches!(i, Instr::Boxed { .. })),
            "the boxed instruction itself must never be removed"
        );
    }

    #[test]
    fn a_boxed_address_is_rewritten_like_any_other_operand() {
        // `each_operand_mut` drives operand rewriting. Skipping `Instr::Boxed` there
        // leaves the address pointing at a value later passes have replaced.
        let old = ValId(7);
        let new = ValId(9);
        let mut b = empty_sb(bid(0x1000), vec![], Terminator::AdoptedReturn);
        b.instrs.push(Instr::Boxed {
            site: 0x1400afec5,
            text: "movups xmm0,[rcx]".into(),
            bytes: vec![0x0F, 0x10, 0x01],
            mem: Some(BoxedMemOp {
                addr: Operand::Val(old),
                bytes: 16,
                writes: false,
            }),
            uses: Vec::new(),
        });

        for ins in &mut b.instrs {
            each_operand_mut(ins, &mut |o| {
                if *o == Operand::Val(old) {
                    *o = Operand::Val(new);
                }
            });
        }

        match &b.instrs[0] {
            Instr::Boxed { mem: Some(m), .. } => {
                assert_eq!(
                    m.addr,
                    Operand::Val(new),
                    "the boxed address was not rewritten"
                );
            }
            other => panic!("expected a boxed instruction, got {other:?}"),
        }
    }

    /// A frame slot every block writes and none reads is dropped everywhere.
    #[test]
    fn a_frame_slot_no_block_reads_is_dropped() {
        let mut blocks = vec![
            empty_sb(bid(1), vec![], Terminator::Jump(bid(2))),
            empty_sb(bid(2), vec![], Terminator::AdoptedReturn),
        ];
        blocks[0].instrs.push(fstore(-0x298, 0x82, Width::W64));
        blocks[1].instrs.push(fstore(-0x298, 0x46, Width::W64));
        // A slot that is read, to show the pass is selective rather than blanket.
        blocks[0].instrs.push(fstore(-0x40, 7, Width::W64));
        blocks[1].instrs.push(Instr::Load {
            dst: ValId(0),
            addr: Operand::Frame(-0x40),
            disp: 0,
            width: Width::W64,
        });

        let escapes = analyze_frame_escapes(&blocks);
        let reads = analyze_frame_reads(&blocks);
        let n = drop_write_only_frame_slots(&mut blocks, &reads, &escapes);
        assert_eq!(n, 2, "both stores to the unread slot should go");
        let left: Vec<i64> = blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .filter_map(|i| match i {
                Instr::Store {
                    addr: Operand::Frame(off),
                    ..
                } => Some(*off),
                _ => None,
            })
            .collect();
        assert_eq!(
            left,
            vec![-0x40],
            "only the read slot's store should remain"
        );
    }

    /// A slot written in one block and read in another survives. The case that makes the pass whole-function rather than per-block.
    #[test]
    fn a_slot_read_by_a_later_block_survives() {
        let mut blocks = vec![
            empty_sb(bid(1), vec![], Terminator::Jump(bid(2))),
            empty_sb(bid(2), vec![], Terminator::AdoptedReturn),
        ];
        blocks[0].instrs.push(fstore(-0x60, 0x1234, Width::W64));
        blocks[1].instrs.push(Instr::Load {
            dst: ValId(0),
            addr: Operand::Frame(-0x60),
            disp: 0,
            width: Width::W64,
        });

        let escapes = analyze_frame_escapes(&blocks);
        let reads = analyze_frame_reads(&blocks);
        let n = drop_write_only_frame_slots(&mut blocks, &reads, &escapes);
        assert_eq!(n, 0, "a cross-block read must keep the store");
    }

    /// A narrow read anywhere in the slot keeps a wide store to it.
    ///
    /// Reads are ranges, not offsets: a one-byte load from the middle of an eight-byte
    /// store observes it, so overlap rather than equality has to be the test.
    #[test]
    fn a_partial_read_keeps_the_whole_store() {
        let mut blocks = vec![empty_sb(bid(1), vec![], Terminator::AdoptedReturn)];
        blocks[0]
            .instrs
            .push(fstore(-0x80, 0xdead_beef, Width::W64));
        blocks[0].instrs.push(Instr::Load {
            dst: ValId(0),
            addr: Operand::Frame(-0x80),
            disp: 4,
            width: Width::W8,
        });

        let escapes = analyze_frame_escapes(&blocks);
        let reads = analyze_frame_reads(&blocks);
        let n = drop_write_only_frame_slots(&mut blocks, &reads, &escapes);
        assert_eq!(n, 0, "a byte read inside the stored range must keep it");
    }

    /// A slot whose address escapes survives even though nothing here reads it. An out-parameter: the function hands the callee a pointer and never loads the slot itself.
    #[test]
    fn an_escaping_slot_survives_with_no_read() {
        let mut blocks = vec![empty_sb(bid(1), vec![], Terminator::AdoptedReturn)];
        blocks[0].instrs.push(fstore(-0x50, 0, Width::W64));
        blocks[0].instrs.push(Instr::Call {
            target: CallTarget::Direct(0x1400_1000),
            args: vec![(Reg::Rcx, Operand::Frame(-0x50))],
        });

        let escapes = analyze_frame_escapes(&blocks);
        let reads = analyze_frame_reads(&blocks);
        let n = drop_write_only_frame_slots(&mut blocks, &reads, &escapes);
        assert_eq!(n, 0, "a slot handed to a callee must keep its store");
    }

    /// Stores at and above the return address are never dropped. These offsets
    /// belong to the caller and may be read after this function returns.
    #[test]
    fn the_callers_slots_are_never_dropped() {
        let mut blocks = vec![empty_sb(bid(1), vec![], Terminator::AdoptedReturn)];
        blocks[0].instrs.push(fstore(0x10, 0, Width::W64));
        blocks[0].instrs.push(fstore(0x8, 0, Width::W64));

        let escapes = analyze_frame_escapes(&blocks);
        let reads = analyze_frame_reads(&blocks);
        let n = drop_write_only_frame_slots(&mut blocks, &reads, &escapes);
        assert_eq!(n, 0, "caller-visible slots must survive");
    }

    /// A boxed instruction's memory operand counts as a read. The IR does not record whether a boxed operand is read, written, or both, so it has to be assumed readable.
    #[test]
    fn a_boxed_memory_operand_counts_as_a_read() {
        let mut blocks = vec![empty_sb(bid(1), vec![], Terminator::AdoptedReturn)];
        blocks[0].instrs.push(fstore(-0x100, 0xabcd, Width::W64));
        blocks[0].instrs.push(Instr::Boxed {
            site: 0x1400_2000,
            bytes: vec![0x90],
            text: "nop".into(),
            mem: Some(BoxedMemOp {
                addr: Operand::Frame(-0x100),
                bytes: 8,
                writes: false,
            }),
            uses: Vec::new(),
        });

        let escapes = analyze_frame_escapes(&blocks);
        let reads = analyze_frame_reads(&blocks);
        let n = drop_write_only_frame_slots(&mut blocks, &reads, &escapes);
        assert_eq!(n, 0, "a boxed operand may read the slot");
    }

    #[test]
    fn a_store_narrows_to_only_the_bytes_that_survive() {
        fn replay(b: &SchedBlock) -> HashMap<i64, u8> {
            let mut mem: HashMap<i64, u8> = HashMap::new();
            for ins in &b.instrs {
                if let Instr::Store {
                    addr: Operand::Frame(off),
                    disp,
                    value: Operand::Imm(v),
                    width,
                } = ins
                {
                    for k in 0..width.bytes() as i64 {
                        mem.insert(off + disp + k, (v >> (8 * k)) as u8);
                    }
                }
            }
            mem
        }

        let mut b = empty_sb(bid(1), vec![], Terminator::AdoptedReturn);
        b.instrs.push(fstore(-0x38, 0x032E_3345, Width::W32));
        b.instrs.push(fstore(-0x37, 0x76, Width::W8));
        b.instrs.push(fstore(-0x36, 0x6b, Width::W8));
        b.instrs.push(fstore(-0x35, 0x46, Width::W8));

        let before = replay(&b);
        let n = narrow_partially_dead_stores(&mut b);
        let after = replay(&b);

        assert_eq!(n, 1, "exactly the dword store should narrow");
        assert_eq!(before, after, "narrowing changed what memory holds");
        assert_eq!(before.get(&-0x38), Some(&0x45u8), "the key byte survives");

        match &b.instrs[0] {
            Instr::Store { width, value, .. } => {
                assert_eq!(*width, Width::W8, "should be a byte store");
                assert_eq!(
                    *value,
                    Operand::Imm(0x45),
                    "the immediate should be truncated to the key byte"
                );
            }
            other => panic!("expected a store, got {other:?}"),
        }
    }

    /// A byte someone reads is never narrowed away.
    #[test]
    fn a_store_is_not_narrowed_across_a_read_or_a_call() {
        // A load of the full dword freezes all four bytes, so nothing may narrow even
        // though later byte stores overwrite three of them.
        let mut b = empty_sb(bid(1), vec![], Terminator::AdoptedReturn);
        b.instrs.push(fstore(-0x38, 0x032E_3345, Width::W32));
        b.instrs.push(Instr::Load {
            dst: ValId(1),
            addr: Operand::Frame(-0x38),
            disp: 0,
            width: Width::W32,
        });
        b.instrs.push(fstore(-0x37, 0x76, Width::W8));
        b.instrs.push(fstore(-0x36, 0x6b, Width::W8));
        b.instrs.push(fstore(-0x35, 0x46, Width::W8));
        assert_eq!(
            narrow_partially_dead_stores(&mut b),
            0,
            "a read of the span must block narrowing"
        );

        // A call is a barrier for the same reason: the callee is handed a pointer to
        // this buffer and reads it.
        let mut b = empty_sb(bid(1), vec![], Terminator::AdoptedReturn);
        b.instrs.push(fstore(-0x38, 0x032E_3345, Width::W32));
        b.instrs.push(Instr::Call {
            target: CallTarget::Direct(0x1_4000_5744),
            args: vec![],
        });
        b.instrs.push(fstore(-0x37, 0x76, Width::W8));
        b.instrs.push(fstore(-0x36, 0x6b, Width::W8));
        b.instrs.push(fstore(-0x35, 0x46, Width::W8));
        assert_eq!(
            narrow_partially_dead_stores(&mut b),
            0,
            "a call must block narrowing"
        );
    }

    /// Consecutive stores to the same frame slot are dead when no read occurs between
    /// them.
    #[test]
    fn a_frame_store_clobbered_before_any_read_is_dead() {
        let mut arena = Arena::new();
        let dest = arena.constant(0, Width::W64);
        let mut sb = empty_sb(bid(0x1000), Vec::new(), Terminator::Return { dest });
        let slot = |v| Instr::Store {
            addr: Operand::Frame(-0x50),
            value: Operand::Imm(v),
            disp: 0,
            width: Width::W64,
        };
        sb.instrs = vec![slot(0xaa), slot(0xbb), slot(0xcc)];
        eliminate_dead(&mut sb);

        let stores: Vec<u64> = sb
            .instrs
            .iter()
            .filter_map(|i| match i {
                Instr::Store {
                    value: Operand::Imm(v),
                    ..
                } => Some(*v),
                _ => None,
            })
            .collect();
        assert_eq!(
            stores,
            vec![0xcc],
            "only the last store to the slot is live; the first two are overwritten              before any read"
        );
    }

    /// A callee handed a pointer into the bytes being re-anchored keeps them where they are.
    #[test]
    fn stores_are_not_re_anchored_when_the_callee_is_given_their_address() {
        let stack_base = STACK_BASE;
        // RSP at the call, as measured for both functions.
        let call_rsp = stack_base.wrapping_sub(0x58);

        let store_at = |off: i64| Instr::Store {
            addr: Operand::Frame(off),
            value: Operand::Imm(0x41),
            disp: 0,
            width: Width::W8,
        };

        // The decryptor: rcx points at -0x36, inside the -0x38..-0x2c span.
        let mut decryptor = vec![
            store_at(-0x38),
            store_at(-0x32),
            store_at(-0x2c),
            Instr::Call {
                target: CallTarget::Direct(0x140005744),
                args: vec![(Reg::Rcx, Operand::Frame(-0x36))],
            },
        ];
        retarget_stack_args(&mut decryptor, &[(3, Some(call_rsp))], stack_base);
        for (n, ins) in decryptor[..3].iter().enumerate() {
            assert!(
                matches!(
                    ins,
                    Instr::Store {
                        addr: Operand::Frame(_),
                        ..
                    }
                ),
                "store {n} feeds a buffer the callee is handed a pointer to, so it must \
                 stay frame-anchored; got {ins:?}"
            );
        }

        let mut io_create = vec![
            store_at(-0x38),
            store_at(-0x30),
            store_at(-0x28),
            Instr::Call {
                target: CallTarget::Direct(0x14002a288),
                args: vec![
                    (Reg::Rcx, Operand::Entry(Reg::Rcx)),
                    (Reg::R8, Operand::Frame(-0x18)),
                ],
            },
        ];
        retarget_stack_args(&mut io_create, &[(3, Some(call_rsp))], stack_base);
        for (n, ins) in io_create[..3].iter().enumerate() {
            assert!(
                matches!(
                    ins,
                    Instr::Store {
                        addr: Operand::OutArg(_),
                        ..
                    }
                ),
                "store {n} is a genuine stack argument and must be re-anchored to the \
                 ABI area; got {ins:?}"
            );
        }
    }

    #[test]
    fn empty_block_chains_are_threaded_and_orphans_dropped() {
        let rsp_only = || vec![(Reg::Rsp, Operand::Imm(0x58))];
        // entry -> a -> b -> real, where a and b do nothing.
        let blocks = vec![
            empty_sb(bid(0x10), Vec::new(), Terminator::Jump(bid(0x20))),
            empty_sb(bid(0x20), rsp_only(), Terminator::Jump(bid(0x30))),
            empty_sb(bid(0x30), rsp_only(), Terminator::Jump(bid(0x40))),
            empty_sb(bid(0x40), Vec::new(), Terminator::AdoptedReturn),
        ];
        let mut threaded = blocks.clone();
        thread_empty_jumps(&mut threaded);
        assert!(
            matches!(threaded[0].terminator, Terminator::Jump(t) if t == bid(0x40)),
            "the entry must jump straight past both empty blocks, got {:?}",
            threaded[0].terminator
        );
        let kept = drop_unreachable(threaded);
        assert_eq!(
            kept.iter().map(|b| b.id).collect::<Vec<_>>(),
            vec![bid(0x10), bid(0x40)],
            "the bypassed blocks have no predecessors left and must be dropped"
        );
    }

    /// A block whose exit actually moves a value must not be threaded past.
    ///
    /// `rbx = Entry(Rdx)` is a real register move the successor depends on. Only an
    /// identity exit, or one naming RSP (which codegen strips), is safe to skip.
    #[test]
    fn a_block_with_a_real_exit_move_is_not_threaded() {
        let mut blocks = vec![
            empty_sb(bid(0x10), Vec::new(), Terminator::Jump(bid(0x20))),
            empty_sb(
                bid(0x20),
                vec![(Reg::Rbx, Operand::Entry(Reg::Rdx))],
                Terminator::Jump(bid(0x30)),
            ),
            empty_sb(bid(0x30), Vec::new(), Terminator::AdoptedReturn),
        ];
        thread_empty_jumps(&mut blocks);
        assert!(
            matches!(blocks[0].terminator, Terminator::Jump(t) if t == bid(0x20)),
            "the exit move has to happen, so the block cannot be skipped, got {:?}",
            blocks[0].terminator
        );
    }

    /// Threading must terminate even if the empty blocks form a cycle.
    ///
    /// Such a graph is an infinite loop and is rejected upstream, but the resolver
    /// walks a chain and must not be the thing that hangs.
    #[test]
    fn threading_a_cycle_of_empty_blocks_terminates() {
        let rsp_only = || vec![(Reg::Rsp, Operand::Imm(0x58))];
        let mut blocks = vec![
            empty_sb(bid(0x10), Vec::new(), Terminator::Jump(bid(0x20))),
            empty_sb(bid(0x20), rsp_only(), Terminator::Jump(bid(0x30))),
            empty_sb(bid(0x30), rsp_only(), Terminator::Jump(bid(0x20))),
        ];
        thread_empty_jumps(&mut blocks);
        // Whichever end it settles on, it must have stopped.
        assert!(matches!(blocks[0].terminator, Terminator::Jump(_)));
    }

    #[test]
    fn a_returns_identity_exit_does_not_pin_a_non_volatile() {
        let mut arena = Arena::default();
        let rcx = arena.init_reg(Reg::Rcx);
        let rsp = arena.init_reg(Reg::Rsp);
        let rbp = arena.init_reg(Reg::Rbp);
        let rbx = arena.init_reg(Reg::Rbx);
        let rax = arena.init_reg(Reg::Rax);

        let regs = vec![
            (Reg::Rcx, rcx),
            (Reg::Rsp, rsp),
            (Reg::Rbp, rbp),
            (Reg::Rbx, rbx),
            (Reg::Rax, rax),
        ];
        let mk = |h: u64, preds: Vec<BlockId>, term: Terminator| Block {
            id: bid(h),
            block_ref: BlockRef(h as u32),
            params: Vec::new(),
            events: Vec::new(),
            from_vm_context: true,
            exit_regs: regs.clone(),
            terminator: term,
            cost: 0,
            preds,
        };

        let cfg = Cfg {
            entry: bid(0),
            arena,
            blocks: vec![
                mk(0, vec![], Terminator::Jump(bid(1))),
                mk(1, vec![bid(0)], Terminator::Return { dest: rsp }),
            ],
            total_steps: 0,
            unresolved: 0,
            back_edges: Vec::new(),
            trivial_phis_removed: 0,
            timed_out: false,
            stack_base: STACK_BASE,
            vm_range: (0, 0),
        };

        // An exit is `(destination, source)`, and the prune keeps or drops it by its
        // *destination*. So block 0 has to name every register it might have to
        // deliver, and the return names what it does with them.
        let block0_exits = vec![
            (Reg::Rcx, Operand::Entry(Reg::Rcx)),
            (Reg::Rbp, Operand::Entry(Reg::Rbp)),
            (Reg::Rbx, Operand::Entry(Reg::Rbx)),
            (Reg::Rax, Operand::Entry(Reg::Rax)),
            (Reg::Rsp, Operand::Entry(Reg::Rsp)),
        ];
        let return_exits = vec![
            // Identity, non-volatile: the epilogue restores it, so it is redundant and
            // must not make RBP live-out at block 0.
            (Reg::Rbp, Operand::Entry(Reg::Rbp)),
            // Not the identity: a real move, so its *source* RCX must reach the return.
            (Reg::Rbx, Operand::Entry(Reg::Rcx)),
            // Volatile: nothing restores it, so an identity exit here is real.
            (Reg::Rax, Operand::Entry(Reg::Rax)),
            (Reg::Rsp, Operand::Entry(Reg::Rsp)),
        ];
        let mut blocks: Vec<SchedBlock> = cfg
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| SchedBlock {
                id: b.id,
                instrs: Vec::new(),
                exits: if i == 0 {
                    block0_exits.clone()
                } else {
                    return_exits.clone()
                },
                callee_set: Vec::new(),
                control: None,
                terminator: b.terminator.clone(),
                fused_cmp: None,
            })
            .collect();

        prune_dead_exits(&cfg, &mut blocks);

        let has = |b: &SchedBlock, r: Reg| b.exits.iter().any(|(x, _)| *x == r);

        assert!(
            !has(&blocks[0], Reg::Rbp),
            "block 0 must not carry RBP: the return's exit for it is the identity and \
             the epilogue restores it, so nothing downstream needs it delivered"
        );
        assert!(
            has(&blocks[0], Reg::Rcx),
            "block 0 must keep RCX: the return moves it into RBX, which is a real move"
        );
        assert!(
            has(&blocks[0], Reg::Rax),
            "block 0 must keep RAX: it is volatile, so no epilogue restores it"
        );
    }

    #[test]
    fn liveness_carries_through_a_block_that_does_not_read() {
        let mut arena = Arena::default();
        let rcx = arena.init_reg(Reg::Rcx);
        let rsp = arena.init_reg(Reg::Rsp);
        let rdi = arena.init_reg(Reg::Rdi);

        // Every block passes RCX, RSP and RDI along unchanged.
        let regs = vec![(Reg::Rcx, rcx), (Reg::Rsp, rsp), (Reg::Rdi, rdi)];
        let mk = |h: u64, preds: Vec<BlockId>, term: Terminator| Block {
            id: bid(h),
            block_ref: BlockRef(h as u32),
            params: Vec::new(),
            events: Vec::new(),
            from_vm_context: true,
            exit_regs: regs.clone(),
            terminator: term,
            cost: 0,
            preds,
        };

        let cfg = Cfg {
            entry: bid(0),
            arena,
            blocks: vec![
                mk(0, vec![], Terminator::Jump(bid(1))),
                mk(1, vec![bid(0), bid(2)], Terminator::Jump(bid(2))),
                // Loops back rather than returning. A return or tail call is an
                // opaque exit that holds every register live, which would mask the
                // pruning this test is checking.
                mk(2, vec![bid(1)], Terminator::Backedge { target: bid(1) }),
            ],
            total_steps: 0,
            unresolved: 0,
            back_edges: Vec::new(),
            trivial_phis_removed: 0,
            timed_out: false,
            stack_base: STACK_BASE,
            vm_range: (0, 0),
        };

        let mut blocks: Vec<SchedBlock> = cfg
            .blocks
            .iter()
            .map(|b| SchedBlock {
                id: b.id,
                instrs: Vec::new(),
                exits: vec![
                    (Reg::Rcx, Operand::Entry(Reg::Rcx)),
                    (Reg::Rsp, Operand::Entry(Reg::Rsp)),
                    (Reg::Rdi, Operand::Entry(Reg::Rdi)),
                ],
                callee_set: Vec::new(),
                control: None,
                terminator: b.terminator.clone(),
                fused_cmp: None,
            })
            .collect();

        // Only block 2 reads RCX, and it reads nothing else.
        blocks[2].instrs.push(Instr::Store {
            addr: Operand::Entry(Reg::Rcx),
            disp: 0,
            value: Operand::Imm(1),
            width: Width::W64,
        });

        prune_dead_exits(&cfg, &mut blocks);

        let has = |b: &SchedBlock, r: Reg| b.exits.iter().any(|(x, _)| *x == r);

        assert!(
            has(&blocks[0], Reg::Rcx),
            "block 0 must keep RCX: block 2 reads it, two edges away"
        );
        assert!(
            has(&blocks[1], Reg::Rcx),
            "block 1 must keep RCX even though it never reads it, because it is on \
             the path to a block that does"
        );
        // RDI is read by nothing anywhere in the loop, so every block drops it.
        for (i, b) in blocks.iter().enumerate() {
            assert!(
                !has(b, Reg::Rdi),
                "nothing reads RDI, so block {i} drops it"
            );
        }
        // Guest RSP is always kept: frame recovery reads it.
        assert!(has(&blocks[0], Reg::Rsp), "RSP is never pruned");
    }
    #[test]
    fn entry_read_follows_the_register_the_value_was_saved_in() {
        let mut arena = Arena::default();
        let in_rdx = arena.init_reg(Reg::Rdx);
        let in_rsp = arena.init_reg(Reg::Rsp);
        let trashed = arena.opaque("clobbered", crate::ir::expr::Width::W64);

        // Block 0 saves the incoming RDX into RBX, and a call destroys RDX itself.
        let b0_exits = vec![(Reg::Rbx, in_rdx), (Reg::Rdx, trashed), (Reg::Rsp, in_rsp)];
        // Block 1 changes nothing.
        let b1_exits = b0_exits.clone();

        let mk = |h: u64, preds: Vec<BlockId>, term: Terminator, ex: Vec<(Reg, Ref)>| Block {
            id: bid(h),
            block_ref: BlockRef(h as u32),
            params: Vec::new(),
            events: Vec::new(),
            from_vm_context: true,
            exit_regs: ex,
            terminator: term,
            cost: 0,
            preds,
        };

        let cfg = Cfg {
            entry: bid(0),
            arena,
            // Order the consumer before its predecessor to keep the pass independent
            // of block ordering.
            blocks: vec![
                mk(0, vec![], Terminator::Jump(bid(1)), b0_exits),
                mk(
                    1,
                    vec![bid(0)],
                    Terminator::Backedge { target: bid(1) },
                    b1_exits,
                ),
            ],
            total_steps: 0,
            unresolved: 0,
            back_edges: Vec::new(),
            trivial_phis_removed: 0,
            timed_out: false,
            stack_base: STACK_BASE,
            vm_range: (0, 0),
        };

        let mut blocks: Vec<SchedBlock> = cfg
            .blocks
            .iter()
            .map(|b| SchedBlock {
                id: b.id,
                instrs: Vec::new(),
                exits: vec![(Reg::Rsp, Operand::Entry(Reg::Rsp))],
                callee_set: Vec::new(),
                control: None,
                terminator: b.terminator.clone(),
                fused_cmp: None,
            })
            .collect();

        // Block 1 passes the incoming RDX as an argument.
        blocks[1].instrs.push(Instr::Call {
            target: CallTarget::Direct(0x2000),
            args: vec![(Reg::Rcx, Operand::Entry(Reg::Rdx))],
        });

        rename_entry_reads(&cfg, &mut blocks);

        let Instr::Call { args, .. } = &blocks[1].instrs[0] else {
            panic!("still a call")
        };
        assert_eq!(
            args[0].1,
            Operand::Entry(Reg::Rbx),
            "the incoming RDX was saved into RBX, so the argument must read RBX, not \
             the RDX the call destroyed"
        );

        // Block 0 defines the value rather than receiving it, so it is left alone.
        assert!(blocks[0].instrs.is_empty());
    }

    /// A narrowed save cannot answer a read wider than it preserves. The other half of the rule above.
    #[test]
    fn a_narrowed_save_is_not_a_home_for_a_wide_read() {
        let mut arena = Arena::default();
        let in_rcx = arena.init_reg(Reg::Rcx);
        let in_rsp = arena.init_reg(Reg::Rsp);
        let mask = arena.constant(0xffff_ffff, crate::ir::expr::Width::W64);
        let narrowed = arena.bin(BinOp::And, in_rcx, mask);
        let addr = arena.constant(0x7fff_fffe_fd14, crate::ir::expr::Width::W64);

        let b0_exits = vec![(Reg::Rcx, addr), (Reg::Rdi, narrowed), (Reg::Rsp, in_rsp)];

        let mk = |h: u64, preds: Vec<BlockId>, term: Terminator, ex: Vec<(Reg, Ref)>| Block {
            id: bid(h),
            block_ref: BlockRef(h as u32),
            params: Vec::new(),
            events: Vec::new(),
            from_vm_context: false,
            exit_regs: ex,
            terminator: term,
            cost: 0,
            preds,
        };

        let cfg = Cfg {
            entry: bid(0),
            arena,
            blocks: vec![
                mk(0, vec![], Terminator::Jump(bid(1)), b0_exits.clone()),
                mk(1, vec![bid(0)], Terminator::AdoptedReturn, b0_exits),
            ],
            total_steps: 0,
            unresolved: 0,
            back_edges: Vec::new(),
            trivial_phis_removed: 0,
            timed_out: false,
            stack_base: STACK_BASE,
            vm_range: (0, 0),
        };

        let mut blocks: Vec<SchedBlock> = cfg
            .blocks
            .iter()
            .map(|b| SchedBlock {
                id: b.id,
                instrs: Vec::new(),
                exits: vec![(Reg::Rsp, Operand::Entry(Reg::Rsp))],
                callee_set: Vec::new(),
                control: None,
                terminator: b.terminator.clone(),
                fused_cmp: None,
            })
            .collect();

        // Block 1 uses the incoming RCX as a 64-bit pointer.
        blocks[1].instrs.push(Instr::Load {
            dst: ValId(0),
            addr: Operand::Entry(Reg::Rcx),
            disp: 0,
            width: Width::W64,
        });

        rename_entry_reads(&cfg, &mut blocks);

        let Instr::Load { addr, .. } = &blocks[1].instrs[0] else {
            panic!("still a load")
        };
        assert_eq!(
            *addr,
            Operand::Entry(Reg::Rcx),
            "RDI keeps only 32 bits, so it cannot stand in for a 64-bit read and the \
             rename must decline rather than substitute a zero-extended value"
        );
    }

    #[test]
    fn a_call_return_value_is_placed_at_the_call() {
        let mut arena = crate::ir::expr::Arena::new();
        let stack_base = 0x7ff0_0000_u64;
        let ret = arena.opaque("call_ret", Width::W64);
        // The reader is a load *through* the returned pointer, so scheduling it
        // requires the return value: exactly the shape of `v3 = load [v2]`.
        let loaded = arena.load(ret, Width::W64);
        let dest = arena.constant(0, Width::W64);
        let events = vec![
            Event::Call {
                target: None,
                site: 0x140001010,
                rsp: Some(stack_base - 0x28),
                import_slot: Some(0x14002a288),
                args: Vec::new(),
                ret: Some(ret),
            },
            Event::Load {
                addr: ret,
                value: loaded,
                width: Width::W64,
                site: 0x140001020,
                region: Region::Guest,
            },
        ];
        let block = Block {
            id: bid(0x1000),
            block_ref: crate::ir::expr::BlockRef(0),
            params: Vec::new(),
            events,
            from_vm_context: true,
            exit_regs: vec![(Reg::Rbx, loaded)],
            terminator: Terminator::Return { dest },
            cost: 0,
            preds: Vec::new(),
        };
        let cfg = Cfg {
            entry: bid(0x1000),
            arena,
            blocks: vec![block],
            total_steps: 0,
            unresolved: 0,
            back_edges: Vec::new(),
            trivial_phis_removed: 0,
            timed_out: false,
            stack_base,
            vm_range: (0, 0),
        };
        let sb = schedule_one(&cfg, &cfg.blocks[0]);
        let call = sb
            .instrs
            .iter()
            .position(|i| matches!(i, Instr::Call { .. }))
            .unwrap_or_else(|| panic!("no call scheduled: {:?}", sb.instrs));
        let opaque = sb
            .instrs
            .iter()
            .position(|i| {
                matches!(
                    i,
                    Instr::Opaque {
                        tag: "call_ret",
                        ..
                    }
                )
            })
            .unwrap_or_else(|| panic!("no return opaque scheduled: {:?}", sb.instrs));
        assert_eq!(
            opaque,
            call + 1,
            "the return opaque must sit immediately after its call, where the value              actually is; got call at {call} and opaque at {opaque} in {:?}",
            sb.instrs
        );
    }

    /// A return value from a *predecessor's* call is read as RAX's entry value. Block boundaries are fixed: a successor reads guest register `R` from `R`.
    #[test]
    fn a_return_value_from_a_predecessors_call_reads_rax_at_entry() {
        let mut arena = crate::ir::expr::Arena::new();
        let stack_base = 0x7ff0_0000_u64;
        // No `Event::Call` in this block: the opaque came from a predecessor.
        let ret = arena.opaque("call_ret", Width::W64);
        let loaded = arena.load(ret, Width::W64);
        let dest = arena.constant(0, Width::W64);
        let events = vec![Event::Load {
            addr: ret,
            value: loaded,
            width: Width::W64,
            site: 0x140001020,
            region: Region::Guest,
        }];
        let block = Block {
            id: bid(0x1000),
            block_ref: crate::ir::expr::BlockRef(0),
            params: Vec::new(),
            events,
            from_vm_context: true,
            exit_regs: vec![(Reg::Rbx, loaded)],
            terminator: Terminator::Return { dest },
            cost: 0,
            preds: Vec::new(),
        };
        let cfg = Cfg {
            entry: bid(0x1000),
            arena,
            blocks: vec![block],
            total_steps: 0,
            unresolved: 0,
            back_edges: Vec::new(),
            trivial_phis_removed: 0,
            timed_out: false,
            stack_base,
            vm_range: (0, 0),
        };
        let sb = schedule_one(&cfg, &cfg.blocks[0]);
        assert!(
            !sb.instrs.iter().any(|i| matches!(
                i,
                Instr::Opaque {
                    tag: "call_ret",
                    ..
                }
            )),
            "no call in this block returns anything, so there is no definition to              place here; got {:?}",
            sb.instrs
        );
        let addr = sb
            .instrs
            .iter()
            .find_map(|i| match i {
                Instr::Load { addr, .. } => Some(*addr),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no load scheduled: {:?}", sb.instrs));
        assert_eq!(
            addr,
            Operand::Entry(Reg::Rax),
            "the pointer arrives in rax at entry, so the load must address it there;              got {addr:?}"
        );
    }

    #[test]
    fn a_boxed_implicit_read_keeps_its_definition_alive() {
        let base = ValId(0);
        let index = ValId(1);
        let mut b = empty_sb(bid(0x1000), vec![], Terminator::AdoptedReturn);
        b.instrs.push(Instr::Opaque {
            dst: base,
            tag: "call_ret",
            width: Width::W64,
            at: Some(Reg::Rax),
        });
        b.instrs.push(Instr::Bin {
            dst: index,
            op: BinOp::Add,
            a: Operand::Val(base),
            b: Operand::Imm(0x6a2),
            width: Width::W64,
            operand_width: Width::W64,
        });
        // `rdmsr` = 0f 32, reading its MSR index from ECX and naming no operand.
        b.instrs.push(Instr::Boxed {
            site: 0x14010e025,
            text: "rdmsr".into(),
            bytes: vec![0x0F, 0x32],
            mem: None,
            uses: vec![(Reg::Rcx, Operand::Val(index))],
        });

        eliminate_dead(&mut b);

        assert!(
            b.instrs
                .iter()
                .any(|i| matches!(i, Instr::Bin { dst, .. } if *dst == index)),
            "the Add computing the MSR index was deleted as dead, so codegen has no \
             location for it and the whole function fails to lower"
        );
    }

    /// A `uses` operand is rewritten like any other.
    ///
    /// `each_operand_mut` drives operand renaming, so leaving `uses` out of it would
    /// point an implicit read at a value later passes have replaced; the same defect
    /// `a_boxed_address_is_rewritten_like_any_other_operand` guards for the address.
    #[test]
    fn a_boxed_implicit_read_is_rewritten_like_any_other_operand() {
        let old = ValId(7);
        let new = ValId(9);
        let mut b = empty_sb(bid(0x1000), vec![], Terminator::AdoptedReturn);
        b.instrs.push(Instr::Boxed {
            site: 0x14010e025,
            text: "rdmsr".into(),
            bytes: vec![0x0F, 0x32],
            mem: None,
            uses: vec![(Reg::Rcx, Operand::Val(old))],
        });
        each_operand_mut(&mut b.instrs[0], &mut |o| {
            if *o == Operand::Val(old) {
                *o = Operand::Val(new);
            }
        });
        let Instr::Boxed { uses, .. } = &b.instrs[0] else {
            panic!("not boxed")
        };
        assert_eq!(
            uses[0].1,
            Operand::Val(new),
            "a uses operand must be renamed with the rest"
        );
    }
}
