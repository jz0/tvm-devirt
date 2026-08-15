//! Register allocation over scheduled blocks. 

use iced_x86::{Decoder, DecoderOptions, InstructionInfoFactory, OpAccess};
use std::collections::{HashMap, HashSet};

use crate::ir::expr::Reg;
use crate::ir::sched::{CallTarget, CastKind, Instr, Operand, SchedBlock, ValId};

/// Registers available to hold intermediate values. RSP is excluded: it is the stack pointer, and the recovered code stores through it.
pub const ALLOCATABLE: [Reg; 15] = [
    Reg::Rax,
    Reg::Rcx,
    Reg::Rdx,
    Reg::Rbx,
    Reg::Rbp,
    Reg::Rsi,
    Reg::Rdi,
    Reg::R8,
    Reg::R9,
    Reg::R10,
    Reg::R11,
    Reg::R12,
    Reg::R13,
    Reg::R14,
    Reg::R15,
];

/// Registers a call may destroy, per the Windows x64 ABI. A value live across a call
/// cannot sit in one of these.
pub const VOLATILE: [Reg; 7] = [
    Reg::Rax,
    Reg::Rcx,
    Reg::Rdx,
    Reg::R8,
    Reg::R9,
    Reg::R10,
    Reg::R11,
];

/// The half-open range of instruction indices over which a value is live. `def` is where it is written, `last_use` the index of its final read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Interval {
    pub val: ValId,
    pub def: usize,
    pub last_use: usize,
    /// Whether the interval spans a call, which rules out the volatile registers.
    pub crosses_call: bool,
    /// Registers destroyed by a boxed instruction the interval spans, as a bitmask over `Reg`. Separate from `crosses_call` because the constraint is narrower. A call clobbers the whole volatile set, so a range crossing one is restricted to the non-volatile pool.
    pub avoid: RegSet,
}

/// A set of registers, as a bitmask over `Reg`'s discriminant.
///
/// A mask rather than a `Vec` so `Interval` and `Range` stay `Copy`, which the
/// allocator's range list relies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RegSet(u16);

impl RegSet {
    pub const EMPTY: RegSet = RegSet(0);

    pub fn insert(&mut self, r: Reg) {
        self.0 |= 1 << (r as u16);
    }

    pub fn contains(&self, r: Reg) -> bool {
        self.0 & (1 << (r as u16)) != 0
    }

    pub fn union(self, other: RegSet) -> RegSet {
        RegSet(self.0 | other.0)
    }

    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }
}

impl Interval {
    /// Whether two intervals need distinct registers.
    ///
    /// The allocator works from the sorted range list rather than asking pairwise, so
    /// only the test that pins down the overlap rule calls this.
    #[cfg(test)]
    pub fn overlaps(&self, other: &Interval) -> bool {
        self.def < other.last_use && other.def < self.last_use
    }
}

/// Whether a live range spans the call at `call`, given the range's extent. `last_use` is one past the last reading position, so a range read *by* the call itself has `last_use == call + 1`.
fn spans_call(def: usize, last_use: usize, call: usize) -> bool {
    def <= call && call + 1 < last_use
}

/// The allocatable registers a boxed instruction destroys, or `None` if it destroys none the allocator manages. A boxed instruction is re-emitted verbatim, so whatever the original encoding wrote it still writes.
fn boxed_clobbers(bytes: &[u8], site: u64) -> RegSet {
    if bytes.is_empty() {
        return RegSet::EMPTY;
    }
    let mut dec = Decoder::with_ip(64, bytes, site, DecoderOptions::NONE);
    let inst = dec.decode();
    if inst.is_invalid() {
        // Cannot tell what it writes, so assume the worst: every allocatable register.
        // A boxed instruction that does not decode also fails to emit, but the
        // allocator runs first and must not hand out a register on a guess.
        let mut all = RegSet::EMPTY;
        for r in ALLOCATABLE {
            all.insert(r);
        }
        return all;
    }
    let mut f = InstructionInfoFactory::new();
    let info = f.info(&inst);
    let mut out = RegSet::EMPTY;
    for u in info.used_registers() {
        let writes = matches!(
            u.access(),
            OpAccess::Write | OpAccess::ReadWrite | OpAccess::CondWrite | OpAccess::ReadCondWrite
        );
        if !writes {
            continue;
        }
        // `decode_reg` maps a sub-register to the full register the allocator manages;
        // `None` means one it does not (segment, control, xmm).
        if let Some((base, _, _)) = crate::ir::lift::decode_reg(u.register())
            && ALLOCATABLE.contains(&base)
        {
            out.insert(base);
        }
    }
    out
}

/// Positions that constrain register choice, and what each rules out. A call clobbers every volatile register, so `None` means "the volatile set".
struct Clobber {
    at: usize,
    /// `None` for a call, which clobbers the whole volatile set. `Some` for a boxed
    /// instruction, naming exactly the registers its encoding writes.
    regs: Option<RegSet>,
}

/// Every position in `b` that destroys registers: calls and boxed instructions.
fn clobber_points(b: &SchedBlock) -> Vec<Clobber> {
    b.instrs
        .iter()
        .enumerate()
        .filter_map(|(i, instr)| match instr {
            Instr::Call { .. } => Some(Clobber { at: i, regs: None }),
            Instr::Boxed {
                bytes, site, uses, ..
            } => {
                // Include both written registers and fixed registers populated for
                // implicit reads.
                let mut regs = boxed_clobbers(bytes, *site);
                for (r, _) in uses {
                    if ALLOCATABLE.contains(r) {
                        regs.insert(*r);
                    }
                }
                // An instruction that writes nothing the allocator manages; `cli`,
                // `invlpg`, an xmm move; constrains nothing.
                if regs.is_empty() {
                    None
                } else {
                    Some(Clobber {
                        at: i,
                        regs: Some(regs),
                    })
                }
            }
            _ => None,
        })
        .collect()
}

/// Live intervals for every value defined in `b`, in definition order.
pub fn intervals(b: &SchedBlock) -> Vec<Interval> {
    let n = b.instrs.len();
    let mut def: HashMap<ValId, usize> = HashMap::new();
    let mut last: HashMap<ValId, usize> = HashMap::new();

    for (i, instr) in b.instrs.iter().enumerate() {
        if let Some(d) = defined(instr) {
            def.insert(d, i);
            // A value is live from its definition even if never read, so that two
            // dead definitions are not given the same register and confused for one.
            last.entry(d).or_insert(i + 1);
        }
        for u in used(instr) {
            last.insert(u, i + 1);
        }
    }

    // Values consumed by the terminator or handed to a successor stay live to the end.
    for u in roots(b) {
        last.insert(u, n + 1);
    }

    // Positions that destroy registers: calls (the whole volatile set) and boxed
    // instructions (only what their encoding writes).
    let clobbers = clobber_points(b);

    let mut out: Vec<Interval> = def
        .iter()
        .map(|(&val, &d)| {
            let lu = last.get(&val).copied().unwrap_or(d + 1);
            // `d < c`, not `d <= c`: a value defined *by* a call is its return value,
            // which the call establishes rather than spans. The same holds for a boxed
            // instruction's results, which are placed at it by the scheduler.
            let spanned = clobbers
                .iter()
                .filter(|c| d < c.at && spans_call(d, lu, c.at));
            let mut crosses_call = false;
            let mut avoid = RegSet::EMPTY;
            for c in spanned {
                match &c.regs {
                    None => crosses_call = true,
                    Some(regs) => avoid = avoid.union(*regs),
                }
            }
            Interval {
                val,
                def: d,
                last_use: lu,
                crosses_call,
                avoid,
            }
        })
        .collect();
    out.sort_by_key(|i| (i.def, i.val.0));
    out
}

/// The value an instruction defines, if any.
fn defined(i: &Instr) -> Option<ValId> {
    match i {
        Instr::Bin { dst, .. }
        | Instr::Un { dst, .. }
        | Instr::Cast { dst, .. }
        | Instr::Select { dst, .. }
        | Instr::Load { dst, .. }
        | Instr::Opaque { dst, .. } => Some(*dst),
        Instr::Store { .. } | Instr::Call { .. } | Instr::Boxed { .. } => None,
    }
}

/// The values an instruction reads.
fn used(i: &Instr) -> Vec<ValId> {
    let mut v = Vec::new();
    let mut push = |o: &Operand| {
        if let Operand::Val(x) = o {
            v.push(*x);
        }
    };
    match i {
        Instr::Bin { a, b, .. } => {
            push(a);
            push(b);
        }
        Instr::Un { a, .. } | Instr::Cast { a, .. } | Instr::Load { addr: a, .. } => push(a),
        Instr::Select { cond, a, b, .. } => {
            push(cond);
            push(a);
            push(b);
        }
        Instr::Store { addr, value, .. } => {
            push(addr);
            push(value);
        }
        Instr::Call { target, args } => {
            if let CallTarget::Indirect(t) = target {
                push(t);
            }
            // Argument values are read by the call and must stay live until it.
            for (_, a) in args {
                push(a);
            }
        }
        // The address of a boxed instruction's memory operand is a real use: it has
        // to be live in a register at that point for codegen to materialize it.
        Instr::Boxed { mem, uses, .. } => {
            if let Some(m) = mem {
                push(&m.addr);
            }
            // The values its implicit register reads consume are uses too: codegen moves
            // them into the registers the encoding reads, so they must be live here. Without
            // this the allocator gives them no location and lowering fails outright.
            for (_, a) in uses {
                push(a);
            }
        }
        Instr::Opaque { .. } => {}
    }
    v
}

/// Values the block needs after its last instruction: exit register values and the
/// branch predicate or jump destination.
fn roots(b: &SchedBlock) -> Vec<ValId> {
    let mut v = Vec::new();
    for (_, o) in &b.exits {
        if let Operand::Val(x) = o {
            v.push(*x);
        }
    }
    // A fused comparison's operands must stay live: the instruction that computed
    // the boolean is gone, but the branch still tests the operands directly.
    if let Some(fc) = &b.fused_cmp {
        if let Operand::Val(x) = &fc.a {
            v.push(*x);
        }
        if let Operand::Val(x) = &fc.b {
            v.push(*x);
        }
    } else if let Some(Operand::Val(x)) = &b.control {
        v.push(*x);
    }
    v
}

/// Where a value physically lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Loc {
    Reg(Reg),
    /// A slot in the reconstructed stack frame.
    Spill(u32),
}

/// A value the allocator must place. Either a computed value or a guest register's
/// entry value, which arrives already in that register and so is pre-coloured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Item {
    Val(ValId),
    /// The value guest register `Reg` held on entry to the block.
    Entry(Reg),
    /// An entry value moved out of its volatile register so it survives a call.
    Saved(Reg),
}

/// A live range over items, so entry values and computed values compete for registers
/// on the same footing.
#[derive(Debug, Clone, Copy)]
pub struct Range {
    pub item: Item,
    pub def: usize,
    pub last_use: usize,
    pub crosses_call: bool,
    /// Registers a boxed instruction in the range's span destroys. See
    /// [`Interval::avoid`].
    pub avoid: RegSet,
    /// Set for entry values: the register the item already occupies and must keep for
    /// as long as it is live, since nothing writes it there.
    pub fixed: Option<Reg>,
}

/// Live ranges for a block, covering both computed values and guest entry values. Entry values start live at position 0 because they are already in their register when the block begins.
pub fn ranges(b: &SchedBlock) -> Vec<Range> {
    let mut out: Vec<Range> = intervals(b)
        .into_iter()
        .map(|i| Range {
            item: Item::Val(i.val),
            def: i.def,
            last_use: i.last_use,
            crosses_call: i.crosses_call,
            avoid: i.avoid,
            fixed: None,
        })
        .collect();

    let clobbers = clobber_points(b);

    // Last read of each guest register's entry value.
    let mut last: HashMap<Reg, usize> = HashMap::new();
    for (i, instr) in b.instrs.iter().enumerate() {
        for r in entry_regs_read(instr) {
            last.insert(r, i + 1);
        }
    }
    for (_, o) in &b.exits {
        if let Operand::Entry(r) | Operand::Param(r) = o {
            last.insert(*r, b.instrs.len() + 1);
        }
    }
    if let Some(Operand::Entry(r) | Operand::Param(r)) = &b.control {
        last.insert(*r, b.instrs.len() + 1);
    }
    if let Some(fc) = &b.fused_cmp {
        for o in [&fc.a, &fc.b] {
            if let Operand::Entry(r) | Operand::Param(r) = o {
                last.insert(*r, b.instrs.len() + 1);
            }
        }
    }

    let mut last: Vec<(Reg, usize)> = last.into_iter().collect();
    last.sort_unstable();
    for (r, lu) in last {
        // An entry value is live from position 0, so `spans_call(0, ..)`; unlike the
        // computed-value case there is no definition to exclude.
        let spanned = clobbers.iter().filter(|c| spans_call(0, lu, c.at));
        let mut crosses_call = false;
        let mut avoid = RegSet::EMPTY;
        for c in spanned {
            match &c.regs {
                None => crosses_call = true,
                Some(regs) => avoid = avoid.union(*regs),
            }
        }
        out.push(Range {
            item: Item::Entry(r),
            def: 0,
            last_use: lu,
            crosses_call,
            avoid,
            fixed: Some(r),
        });
    }
    out.sort_by_key(|r| (r.def, r.last_use));
    out
}

/// Guest registers whose entry value an instruction reads.
fn entry_regs_read(i: &Instr) -> Vec<Reg> {
    let mut v = Vec::new();
    let mut push = |o: &Operand| {
        if let Operand::Entry(r) | Operand::Param(r) = o {
            v.push(*r);
        }
    };
    match i {
        Instr::Bin { a, b, .. } => {
            push(a);
            push(b);
        }
        Instr::Un { a, .. } | Instr::Cast { a, .. } | Instr::Load { addr: a, .. } => push(a),
        Instr::Select { cond, a, b, .. } => {
            push(cond);
            push(a);
            push(b);
        }
        Instr::Store { addr, value, .. } => {
            push(addr);
            push(value);
        }
        Instr::Call { target, args } => {
            if let CallTarget::Indirect(t) = target {
                push(t);
            }
            // Argument values are read by the call and must stay live until it.
            for (_, a) in args {
                push(a);
            }
        }
        // The address of a boxed instruction's memory operand is a real use: it has
        // to be live in a register at that point for codegen to materialize it.
        Instr::Boxed { mem, uses, .. } => {
            if let Some(m) = mem {
                push(&m.addr);
            }
            // The values its implicit register reads consume are uses too: codegen moves
            // them into the registers the encoding reads, so they must be live here. Without
            // this the allocator gives them no location and lowering fails outright.
            for (_, a) in uses {
                push(a);
            }
        }
        Instr::Opaque { .. } => {}
    }
    v
}

/// One value's location over one span of the block. A value does not necessarily live in one place for its whole range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub item: Item,
    pub start: usize,
    pub end: usize,
    pub loc: Loc,
}

/// A copy the emitter must insert, because a value changes location at `at`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Copy_ {
    pub item: Item,
    pub at: usize,
    pub from: Loc,
    pub to: Loc,
}

/// The result of allocating one block.
#[derive(Debug, Clone, Default)]
pub struct Alloc {
    pub segments: Vec<Segment>,
    /// Copies to insert, in increasing `at` order.
    pub copies: Vec<Copy_>,
    /// Number of distinct stack slots the block needs.
    pub slots: u32,
}

impl Alloc {
    /// Where `item` lives at instruction `pos`.
    pub fn loc_at(&self, item: Item, pos: usize) -> Option<Loc> {
        self.segments
            .iter()
            .find(|s| s.item == item && s.start <= pos && pos < s.end)
            .map(|s| s.loc)
    }

    pub fn free_volatile_at(&self, pos: usize, avoid: &[Reg]) -> Option<Reg> {
        self.free_from(&VOLATILE, pos, avoid)
    }

    fn free_from(&self, pool: &[Reg], pos: usize, avoid: &[Reg]) -> Option<Reg> {
        pool.iter()
            .copied()
            .filter(|r| !avoid.contains(r))
            .find(|r| {
                !self
                    .segments
                    .iter()
                    .any(|s| s.loc == Loc::Reg(*r) && s.start <= pos && pos < s.end)
            })
    }
}

/// Assign locations to every live range in `b`.
///
/// Linear scan, with three departures from the textbook version, each forced by
/// something in the recovered code:
///
/// - **Entry values are pre-coloured.** A guest register's entry value is already in
///   that register on entry and nothing puts it there, so it cannot simply be assigned
///   elsewhere.
/// - **Ranges crossing a call avoid the volatile registers.** For a computed value
///   that is just a restricted pool. For a pre-coloured entry value in a volatile
///   register it is impossible, so the range is **split** at the first call it crosses:
///   it stays in its own register up to the call, and a copy moves it somewhere
///   preserved for the remainder.
/// - **Spill slots are reused.** Slots are allocated with the same interference test as
///   registers, so a block with many short-lived spills does not accumulate one slot
///   per spill. Without this a single block wanted 333 slots.
pub fn allocate(b: &SchedBlock) -> Alloc {
    let clobbers = clobber_points(b);

    // Split pre-coloured ranges that outlive a call into a pre-call part that keeps
    // the register and a post-call part that must be placed somewhere preserved.
    let mut work: Vec<Range> = Vec::new();
    let mut pending_copies: Vec<(Item, usize)> = Vec::new();
    for r in ranges(b) {
        match r.fixed {
            // A pre-coloured value in a register something later destroys cannot stay there.
            Some(reg) => {
                // `r.def <= c`, not `<`: an entry value is live *before* the block's first instruction, because the caller put it there.
                let killer = clobbers
                    .iter()
                    .filter(|c| spans_call(r.def, r.last_use, c.at))
                    .find(|c| match &c.regs {
                        None => VOLATILE.contains(&reg),
                        Some(regs) => regs.contains(reg),
                    });
                match killer {
                    Some(c) => {
                        work.push(Range {
                            last_use: c.at,
                            crosses_call: false,
                            ..r
                        });
                        // What the saved half must survive is recomputed over its own span rather than inherited from the split point.
                        let mut saved_crosses = false;
                        let mut saved_avoid = RegSet::EMPTY;
                        for k in clobbers
                            .iter()
                            .filter(|k| spans_call(c.at, r.last_use, k.at))
                        {
                            match &k.regs {
                                None => saved_crosses = true,
                                Some(regs) => saved_avoid = saved_avoid.union(*regs),
                            }
                        }
                        // The clobber it was split at destroys registers at exactly
                        // this position, so it counts too.
                        if let Some(regs) = &c.regs {
                            saved_avoid = saved_avoid.union(*regs);
                        }
                        work.push(Range {
                            item: Item::Saved(reg),
                            def: c.at,
                            last_use: r.last_use,
                            crosses_call: saved_crosses,
                            avoid: saved_avoid,
                            fixed: None,
                        });
                        pending_copies.push((Item::Saved(reg), c.at));
                    }
                    None => work.push(r),
                }
            }
            _ => work.push(r),
        }
    }
    work.sort_by_key(|r| (r.def, r.last_use));

    let mut alloc = Alloc::default();
    let mut busy: HashMap<Reg, Range> = HashMap::new();
    // Spill slot -> the range occupying it, so slots can be reused.
    let mut slot_of: Vec<Option<Range>> = Vec::new();

    for r in work.iter().filter(|r| r.fixed.is_some()) {
        let reg = r.fixed.expect("filtered on fixed");
        busy.insert(reg, *r);
        alloc.segments.push(Segment {
            item: r.item,
            start: r.def,
            end: r.last_use,
            loc: Loc::Reg(reg),
        });
    }

    let take_slot = |r: &Range, slots: &mut Vec<Option<Range>>| -> Loc {
        for (i, occupant) in slots.iter_mut().enumerate() {
            if occupant.is_none_or(|h| h.last_use <= r.def) {
                *occupant = Some(*r);
                return Loc::Spill(i as u32);
            }
        }
        slots.push(Some(*r));
        Loc::Spill(slots.len() as u32 - 1)
    };

    // Coalescing hints: the item each value's defining instruction reads first.
    let hint_src: HashMap<Item, Item> = b
        .instrs
        .iter()
        .filter_map(|i| match i {
            // Only the operations that lower to a two-operand form benefit. `a` is
            // the operand that shares the destination register.
            Instr::Bin { dst, a, .. } | Instr::Un { dst, a, .. } => match a {
                Operand::Val(v) => Some((Item::Val(*dst), Item::Val(*v))),
                _ => None,
            },
            // A truncation to a narrower width is a no-op when source and destination share a register: the emitter already skips it, so the hint turns the copy into nothing at all.
            Instr::Cast {
                dst,
                a,
                kind: CastKind::Trunc,
                ..
            } => match a {
                Operand::Val(v) => Some((Item::Val(*dst), Item::Val(*v))),
                _ => None,
            },
            _ => None,
        })
        .collect();

    // Destination-side coalescing hints: the guest register a value is exited in. The mirror of `hint_src`. Only values are hinted. Sort before the first-wins pass so competing hints resolve deterministically.
    let hint_exit: HashMap<Item, Reg> = {
        let mut claimed: HashSet<Reg> = HashSet::new();
        let mut m: HashMap<Item, Reg> = HashMap::new();
        let mut exits: Vec<(Reg, ValId)> = b
            .exits
            .iter()
            .filter_map(|(r, o)| match o {
                Operand::Val(v) => Some((*r, *v)),
                _ => None,
            })
            .collect();
        exits.sort_unstable();
        for (r, v) in exits {
            // Guest RSP is not an allocated value; the frame layout owns it; so
            // steering a computation into it would corrupt the stack pointer.
            if r == Reg::Rsp {
                continue;
            }
            // One value per register, and one register per value: a second value
            // exiting in the same register cannot also have it.
            if claimed.contains(&r) {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = m.entry(Item::Val(v)) {
                e.insert(r);
                claimed.insert(r);
            }
        }
        m
    };

    for r in work.iter().filter(|r| r.fixed.is_none()) {
        busy.retain(|_, held| held.last_use > r.def);
        let pool: Vec<Reg> = if r.crosses_call {
            non_volatile()
        } else {
            ALLOCATABLE.to_vec()
        };
        // And drop anything a boxed instruction in this range's span destroys. Narrower
        // than the call constraint on purpose: `cpuid` rules out four registers, not
        // the whole volatile set.
        let pool: Vec<Reg> = pool.into_iter().filter(|x| !r.avoid.contains(*x)).collect();
        // Prefer the register the first operand is vacating. `busy` has already had ranges ending at or before this def removed, so a register that held a dying operand now reads as free; the hint just says which free register to take.
        let hinted = hint_src
            .get(&r.item)
            .and_then(|src| {
                // The operand must die into this definition. `last_use` is recorded one past the final reading instruction, so an operand read only by the instruction at `r.def` has `last_use == r.def + 1`.
                let dies_here = work
                    .iter()
                    .find(|w| w.item == *src)
                    .is_some_and(|w| w.last_use <= r.def + 1);
                dies_here.then_some(*src)
            })
            .and_then(|src| alloc.segments.iter().rev().find(|s| s.item == src))
            .and_then(|s| match s.loc {
                Loc::Reg(reg) => Some(reg),
                Loc::Spill(_) => None,
            })
            // `busy` still lists the operand's register, because `retain` above keeps
            // ranges whose `last_use > r.def` and a dying operand's is `r.def + 1`.
            // Allowing exactly that one register through is what makes the copy go
            // away; every other occupied register is still genuinely unavailable.
            .filter(|reg| {
                pool.contains(reg) && busy.get(reg).is_none_or(|held| held.last_use <= r.def + 1)
            });
        // Failing that, the register this value is exited in, if it is free for the whole range. `!busy.contains_key` is the full test here, unlike the source hint's allowance for a dying operand.
        let hinted = hinted.or_else(|| {
            hint_exit
                .get(&r.item)
                .copied()
                .filter(|reg| pool.contains(reg) && !busy.contains_key(reg))
        });
        let loc = match hinted.or_else(|| pool.iter().copied().find(|reg| !busy.contains_key(reg)))
        {
            Some(reg) => {
                busy.insert(reg, *r);
                Loc::Reg(reg)
            }
            None => {
                let victim = pool
                    .iter()
                    .filter_map(|reg| busy.get(reg).map(|h| (*reg, *h)))
                    // A pre-coloured range cannot be evicted: it is only ever in its
                    // own register, so freeing it would need a copy this point in the
                    // scan cannot place. This range spills instead.
                    .filter(|(_, h)| h.fixed.is_none())
                    .max_by_key(|(_, h)| h.last_use);
                match victim {
                    Some((reg, held)) if held.last_use > r.last_use => {
                        // Rewrite the evicted range's segment to the slot it moves to.
                        let slot = take_slot(&held, &mut slot_of);
                        if let Some(seg) = alloc.segments.iter_mut().find(|s| s.item == held.item) {
                            seg.loc = slot;
                        }
                        busy.insert(reg, *r);
                        Loc::Reg(reg)
                    }
                    _ => take_slot(r, &mut slot_of),
                }
            }
        };
        alloc.segments.push(Segment {
            item: r.item,
            start: r.def,
            end: r.last_use,
            loc,
        });
    }

    // Now that every segment has a location, the split copies can be described.
    for (item, at) in pending_copies {
        let Item::Saved(reg) = item else { continue };
        let Some(to) = alloc.loc_at(item, at) else {
            continue;
        };
        alloc.copies.push(Copy_ {
            item,
            at,
            from: Loc::Reg(reg),
            to,
        });
    }
    alloc.copies.sort_by_key(|c| c.at);
    alloc.slots = slot_of.len() as u32;
    alloc
}

/// A move the emitter must make at a block boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    /// `dst <- src`, a register copy or a reload from a spill slot.
    Set { dst: Reg, src: Loc },
    /// `dst <- imm`.
    Imm { dst: Reg, value: u64 },
    /// `dst <- rsp + offset`, materializing a stack address.
    Frame { dst: Reg, offset: i64 },
    /// Exchange two registers, to break a cycle.
    Swap { a: Reg, b: Reg },
}

/// Sequence the parallel copy that places a block's exit values into the guest registers its successors will read them from. The exits are a *parallel* assignment: every destination takes its source's value as it was at the end of the block.
pub fn reconcile(b: &SchedBlock, a: &Alloc) -> Vec<Move> {
    let end = b.instrs.len();
    // Pending moves as (dst, source), dropping the ones already satisfied.
    let mut pending: Vec<(Reg, Src)> = Vec::new();
    for (reg, op) in &b.exits {
        let src = match op {
            Operand::Imm(v) => Src::Imm(*v),
            // A stack address is computed into the destination rather than copied
            // from anywhere, so it is never part of a cycle.
            Operand::Frame(o) => Src::Frame(*o),
            // Only ever a store address, never a value: `retarget_stack_args` rewrites
            // the address of a `Store`, and an exit register holds a value.
            Operand::OutArg(n) => unreachable!("outgoing argument slot {n} as an exit value"),
            Operand::Val(v) => match a.loc_at(Item::Val(*v), end) {
                Some(l) => Src::At(l),
                None => continue,
            },
            Operand::Entry(r) | Operand::Param(r) => {
                // An entry value may have been moved out of its register to survive a
                // call, so its location at the end of the block is what matters, not
                // the register it started in.
                match a
                    .loc_at(Item::Saved(*r), end)
                    .or_else(|| a.loc_at(Item::Entry(*r), end))
                {
                    Some(l) => Src::At(l),
                    None => Src::At(Loc::Reg(*r)),
                }
            }
        };
        // A register already holding what it needs to hold needs no move.
        if src == Src::At(Loc::Reg(*reg)) {
            continue;
        }
        pending.push((*reg, src));
    }

    let mut out = Vec::new();
    while !pending.is_empty() {
        // A destination nobody else reads can be written immediately.
        let ready = pending.iter().position(|(dst, _)| {
            !pending
                .iter()
                .any(|(other, src)| other != dst && *src == Src::At(Loc::Reg(*dst)))
        });
        match ready {
            Some(i) => {
                let (dst, src) = pending.remove(i);
                out.push(match src {
                    Src::Imm(v) => Move::Imm { dst, value: v },
                    Src::Frame(o) => Move::Frame { dst, offset: o },
                    Src::At(l) => Move::Set { dst, src: l },
                });
            }
            None => {
                // Everything left is on a cycle. Break one with a swap.
                let (dst, src) = pending.remove(0);
                let Src::At(Loc::Reg(from)) = src else {
                    // Only a register source can be part of a cycle: a spill slot or
                    // an immediate is not a destination, so nothing reads it.
                    out.push(match src {
                        Src::Imm(v) => Move::Imm { dst, value: v },
                        Src::Frame(o) => Move::Frame { dst, offset: o },
                        Src::At(l) => Move::Set { dst, src: l },
                    });
                    continue;
                };
                out.push(Move::Swap { a: dst, b: from });
                // The old value of `dst` is now in `from`, so redirect readers.
                for (_, s) in pending.iter_mut() {
                    if *s == Src::At(Loc::Reg(dst)) {
                        *s = Src::At(Loc::Reg(from));
                    } else if *s == Src::At(Loc::Reg(from)) {
                        *s = Src::At(Loc::Reg(dst));
                    }
                }
                // The swap satisfied this move, and may have satisfied others.
                pending.retain(|(d, s)| *s != Src::At(Loc::Reg(*d)));
            }
        }
    }
    out
}

/// Render a boundary move as the instruction it becomes.
pub fn render_move(m: &Move) -> String {
    match m {
        Move::Set {
            dst,
            src: Loc::Reg(r),
        } => format!("mov {}, {}", dst.name(), r.name()),
        Move::Set {
            dst,
            src: Loc::Spill(i),
        } => format!("mov {}, [spill{i}]", dst.name()),
        Move::Imm { dst, value } => format!("mov {}, {value:#x}", dst.name()),
        Move::Frame { dst, offset } => {
            format!(
                "lea {}, [rsp{}{:#x}]",
                dst.name(),
                if *offset < 0 { "-" } else { "+" },
                offset.abs()
            )
        }
        Move::Swap { a, b } => format!("xchg {}, {}", a.name(), b.name()),
    }
}

/// Where a boundary move reads from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Src {
    At(Loc),
    Imm(u64),
    /// A stack address, materialized with a `lea` from RSP.
    Frame(i64),
}

/// A violated allocation invariant.
///
/// The payloads are read through `Debug` when `TVM_ALLOC_ERRS` is set, which the
/// dead-code lint cannot see; without them a reported violation would not say which
/// register or value it was about.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum AllocError {
    /// Two ranges that are live at the same time were given one register.
    Conflict { reg: Reg, a: Item, b: Item },
    /// A range live across a call was placed in a register the call destroys.
    VolatileAcrossCall { reg: Reg, item: Item },
    /// A live range received no location at all.
    Unplaced { item: Item },
    /// The boundary move sequence does not realize the block's exit assignment.
    Reconcile { detail: String },
}

/// Check an allocation against the invariants it is supposed to guarantee.
pub fn verify(b: &SchedBlock, a: &Alloc) -> Vec<AllocError> {
    let mut errs = Vec::new();

    // Index segments by item so coverage is checked against a range's own segments
    // rather than by scanning every segment at every position.
    let mut by_item: HashMap<Item, Vec<Segment>> = HashMap::new();
    for s in &a.segments {
        by_item.entry(s.item).or_default().push(*s);
    }
    for v in by_item.values_mut() {
        v.sort_by_key(|s| s.start);
    }

    // Every live range must be covered for its whole extent, so the emitter always
    // has somewhere to read the value from. A split range is covered jointly by its
    // own segments and its `Saved` half.
    for r in ranges(b) {
        let mut segs: Vec<Segment> = by_item.get(&r.item).cloned().unwrap_or_default();
        if let Some(reg) = r.fixed {
            segs.extend(by_item.get(&Item::Saved(reg)).cloned().unwrap_or_default());
        }
        segs.sort_by_key(|s| s.start);
        let mut covered = r.def;
        for s in &segs {
            if s.start > covered {
                break;
            }
            covered = covered.max(s.end);
        }
        if covered < r.last_use {
            errs.push(AllocError::Unplaced { item: r.item });
        }
    }

    // No value may sit in a register something later destroys. A segment ending exactly
    // at the clobber does not span it: the value is dead before it executes.
    let clobbers = clobber_points(b);
    for s in &a.segments {
        if let Loc::Reg(reg) = s.loc {
            let destroyed = clobbers
                .iter()
                .filter(|c| spans_call(s.start, s.end, c.at))
                .any(|c| match &c.regs {
                    None => VOLATILE.contains(&reg),
                    Some(regs) => regs.contains(reg),
                });
            if destroyed {
                errs.push(AllocError::VolatileAcrossCall { reg, item: s.item });
            }
        }
    }

    // No two segments overlapping in time may share a location. Grouped by location,
    // so only genuine candidates are compared.
    let mut by_loc: HashMap<Loc, Vec<Segment>> = HashMap::new();
    for s in &a.segments {
        by_loc.entry(s.loc).or_default().push(*s);
    }
    for (loc, mut group) in by_loc {
        group.sort_by_key(|s| s.start);
        for w in group.windows(2) {
            // `end` is one past the last reading position, so a segment ending at exactly `w[1].start + 1` is read by the very instruction that defines `w[1]`.
            let handoff = w[0].end == w[1].start + 1;
            if w[0].end > w[1].start && !handoff && w[0].item != w[1].item {
                if let Loc::Reg(reg) = loc {
                    errs.push(AllocError::Conflict {
                        reg,
                        a: w[0].item,
                        b: w[1].item,
                    });
                }
            }
        }
    }
    errs
}

/// Registers preserved across a call, and so usable for values live across one.
pub fn non_volatile() -> Vec<Reg> {
    ALLOCATABLE
        .iter()
        .copied()
        .filter(|r| !VOLATILE.contains(r))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::expr::Width;
    use crate::ir::sched::CastKind;
    use crate::ir::{BlockId, Terminator};

    fn blk(instrs: Vec<Instr>, exits: Vec<(Reg, Operand)>) -> SchedBlock {
        SchedBlock {
            id: BlockId {
                handler: 0,
                vip: None,
            },
            instrs,
            exits,
            callee_set: Vec::new(),
            control: None,
            terminator: Terminator::Unresolved {
                reason: String::new(),
            },
            fused_cmp: None,
        }
    }

    fn v(n: u32) -> ValId {
        ValId(n)
    }

    #[test]
    fn interval_ends_at_last_use() {
        // v0 = zext rax ; v1 = zext v0 ; store [v1], v1
        let b = blk(
            vec![
                Instr::Cast {
                    dst: v(0),
                    kind: CastKind::Zext,
                    a: Operand::Entry(Reg::Rax),
                    from: Width::W8,
                    to: Width::W64,
                },
                Instr::Cast {
                    dst: v(1),
                    kind: CastKind::Zext,
                    a: Operand::Val(v(0)),
                    from: Width::W8,
                    to: Width::W64,
                },
                Instr::Store {
                    disp: 0,
                    addr: Operand::Val(v(1)),
                    value: Operand::Val(v(1)),
                    width: Width::W64,
                },
            ],
            Vec::new(),
        );
        let ivs = intervals(&b);
        assert_eq!(
            ivs[0],
            Interval {
                val: v(0),
                def: 0,
                last_use: 2,
                crosses_call: false,
                avoid: RegSet::EMPTY
            }
        );
        assert_eq!(
            ivs[1],
            Interval {
                val: v(1),
                def: 1,
                last_use: 3,
                crosses_call: false,
                avoid: RegSet::EMPTY
            }
        );
        // v0 dies before v1 is last used, but they still overlap: v1 is defined while
        // v0 is live, so they cannot share a register.
        assert!(ivs[0].overlaps(&ivs[1]));
    }

    /// A chain of single-use arithmetic must be allocated to one register, so the two-operand lowering needs no copies at all.
    #[test]
    fn single_use_arithmetic_chain_coalesces_into_one_register() {
        // v0 = rax + 1 ; v1 = v0 | 2 ; v2 = v1 ^ 3 ; exit rax = v2
        let b = blk(
            vec![
                Instr::Bin {
                    dst: v(0),
                    op: crate::ir::expr::BinOp::Add,
                    a: Operand::Entry(Reg::Rax),
                    b: Operand::Imm(1),
                    width: Width::W64,
                    operand_width: Width::W64,
                },
                Instr::Bin {
                    dst: v(1),
                    op: crate::ir::expr::BinOp::Or,
                    a: Operand::Val(v(0)),
                    b: Operand::Imm(2),
                    width: Width::W64,
                    operand_width: Width::W64,
                },
                Instr::Bin {
                    dst: v(2),
                    op: crate::ir::expr::BinOp::Xor,
                    a: Operand::Val(v(1)),
                    b: Operand::Imm(3),
                    width: Width::W64,
                    operand_width: Width::W64,
                },
            ],
            vec![(Reg::Rax, Operand::Val(v(2)))],
        );
        let a = allocate(&b);
        assert!(
            verify(&b, &a).is_empty(),
            "coalescing must not break the allocation"
        );

        // Each value dies into the next, so all three share one register.
        let loc_of = |val: u32, pos: usize| a.loc_at(Item::Val(v(val)), pos);
        let r0 = loc_of(0, 0);
        assert_eq!(r0, loc_of(1, 1), "v1 must reuse the register v0 dies in");
        assert_eq!(r0, loc_of(2, 2), "v2 must reuse it too");
        assert!(
            matches!(r0, Some(Loc::Reg(_))),
            "the chain must stay in a register"
        );
    }

    /// The relaxed overlap rule must accept only the one-position handoff, and still reject a real conflict. `verify` treats `end == start + 1` as a legal share, because `end` is one past the last read and x86 reads the operand before writing the result.
    #[test]
    fn overlap_rule_admits_the_handoff_but_not_a_real_conflict() {
        let b = blk(Vec::new(), Vec::new());
        let seg = |item: Item, start: usize, end: usize, reg: Reg| Segment {
            item,
            start,
            end,
            loc: Loc::Reg(reg),
        };

        // v0 read by the instruction at 1, which defines v1: end == start + 1.
        let handoff = Alloc {
            segments: vec![
                seg(Item::Val(v(0)), 0, 2, Reg::Rax),
                seg(Item::Val(v(1)), 1, 3, Reg::Rax),
            ],
            ..Alloc::default()
        };
        assert!(
            !verify(&b, &handoff)
                .iter()
                .any(|e| matches!(e, AllocError::Conflict { .. })),
            "a read-then-write handoff is not a conflict"
        );

        // One position further and v0 is still live after v1 is defined.
        let real = Alloc {
            segments: vec![
                seg(Item::Val(v(0)), 0, 3, Reg::Rax),
                seg(Item::Val(v(1)), 1, 4, Reg::Rax),
            ],
            ..Alloc::default()
        };
        assert!(
            verify(&b, &real)
                .iter()
                .any(|e| matches!(e, AllocError::Conflict { .. })),
            "a genuine overlap must still be reported"
        );
    }

    #[test]
    fn exit_value_lives_to_end_of_block() {
        let b = blk(
            vec![Instr::Cast {
                dst: v(0),
                kind: CastKind::Zext,
                a: Operand::Entry(Reg::Rax),
                from: Width::W8,
                to: Width::W64,
            }],
            vec![(Reg::Rbx, Operand::Val(v(0)))],
        );
        let ivs = intervals(&b);
        assert_eq!(
            ivs[0].last_use, 2,
            "an exit value must outlive the last instruction"
        );
    }

    #[test]
    fn value_live_across_call_is_flagged() {
        let b = blk(
            vec![
                Instr::Cast {
                    dst: v(0),
                    kind: CastKind::Zext,
                    a: Operand::Entry(Reg::Rax),
                    from: Width::W8,
                    to: Width::W64,
                },
                Instr::Call {
                    target: CallTarget::Direct(0x1000),
                    args: Vec::new(),
                },
                Instr::Store {
                    disp: 0,
                    addr: Operand::Val(v(0)),
                    value: Operand::Imm(0),
                    width: Width::W64,
                },
            ],
            Vec::new(),
        );
        let ivs = intervals(&b);
        assert!(ivs[0].crosses_call);
    }

    #[test]
    fn entry_value_keeps_its_own_register() {
        // v0 is computed from rax, and rax's entry value is still needed afterwards.
        // The allocator must not hand rax to v0.
        let b = blk(
            vec![Instr::Cast {
                dst: v(0),
                kind: CastKind::Zext,
                a: Operand::Entry(Reg::Rax),
                from: Width::W8,
                to: Width::W64,
            }],
            vec![
                (Reg::Rbx, Operand::Entry(Reg::Rax)),
                (Reg::Rcx, Operand::Val(v(0))),
            ],
        );
        let a = allocate(&b);
        assert_eq!(a.loc_at(Item::Entry(Reg::Rax), 0), Some(Loc::Reg(Reg::Rax)));
        assert_ne!(
            a.loc_at(Item::Val(v(0)), 1),
            Some(Loc::Reg(Reg::Rax)),
            "v0 must not take the register still holding rax's entry value"
        );
    }

    #[test]
    fn value_crossing_a_call_gets_a_non_volatile() {
        let b = blk(
            vec![
                Instr::Opaque {
                    dst: v(0),
                    tag: "t",
                    width: Width::W64,
                    at: None,
                },
                Instr::Call {
                    target: CallTarget::Direct(0x1000),
                    args: Vec::new(),
                },
                Instr::Store {
                    disp: 0,
                    addr: Operand::Val(v(0)),
                    value: Operand::Imm(0),
                    width: Width::W64,
                },
            ],
            Vec::new(),
        );
        let a = allocate(&b);
        match a.loc_at(Item::Val(v(0)), 1) {
            Some(Loc::Reg(r)) => assert!(
                !VOLATILE.contains(&r),
                "a value live across a call cannot sit in volatile {r:?}"
            ),
            Some(Loc::Spill(_)) => {}
            None => panic!("v0 was not placed"),
        }
    }

    #[test]
    fn excess_pressure_spills() {
        // More simultaneously-live values than there are registers. Every value must
        // still be placed, with the overflow going to stack slots.
        let n = ALLOCATABLE.len() + 4;
        let mut instrs: Vec<Instr> = (0..n)
            .map(|i| Instr::Opaque {
                dst: v(i as u32),
                tag: "t",
                width: Width::W64,
                at: None,
            })
            .collect();
        // Keep them all live to the end by storing each one.
        for i in 0..n {
            instrs.push(Instr::Store {
                disp: 0,
                addr: Operand::Imm(0x1000),
                value: Operand::Val(v(i as u32)),
                width: Width::W64,
            });
        }
        let b = blk(instrs, Vec::new());
        let a = allocate(&b);
        for i in 0..n {
            assert!(
                a.loc_at(Item::Val(v(i as u32)), n).is_some(),
                "value {i} unplaced"
            );
        }
        assert!(a.slots > 0, "excess pressure must produce spill slots");
        // No two live values may share a register.
        let mut seen = HashSet::new();
        for i in 0..n {
            if let Some(Loc::Reg(r)) = a.loc_at(Item::Val(v(i as u32)), n) {
                assert!(
                    seen.insert(r),
                    "register {r:?} assigned twice among live values"
                );
            }
        }
    }

    #[test]
    fn entry_value_in_volatile_is_split_across_a_call() {
        // rcx's entry value is needed after a call, but a call destroys rcx. The
        // range must split: rcx up to the call, somewhere preserved afterwards, with a
        // copy between.
        let b = blk(
            vec![
                Instr::Store {
                    disp: 0,
                    addr: Operand::Entry(Reg::Rcx),
                    value: Operand::Imm(0),
                    width: Width::W64,
                },
                Instr::Call {
                    target: CallTarget::Direct(0x1000),
                    args: Vec::new(),
                },
                Instr::Store {
                    disp: 0,
                    addr: Operand::Entry(Reg::Rcx),
                    value: Operand::Imm(1),
                    width: Width::W64,
                },
            ],
            Vec::new(),
        );
        let a = allocate(&b);
        assert_eq!(
            a.loc_at(Item::Entry(Reg::Rcx), 0),
            Some(Loc::Reg(Reg::Rcx)),
            "before the call the value is still in rcx"
        );
        let after = a
            .loc_at(Item::Saved(Reg::Rcx), 2)
            .expect("split half must be placed");
        if let Loc::Reg(r) = after {
            assert!(
                !VOLATILE.contains(&r),
                "the saved half must survive the call"
            );
        }
        assert_eq!(a.copies.len(), 1, "the split needs exactly one copy");
        assert_eq!(a.copies[0].from, Loc::Reg(Reg::Rcx));
        assert_eq!(a.copies[0].to, after);
        assert!(verify(&b, &a).is_empty(), "{:?}", verify(&b, &a));
    }

    #[test]
    fn spill_slots_are_reused_when_lifetimes_do_not_overlap() {
        // Two batches of high pressure separated in time. The second batch should
        // reuse the first batch's slots rather than allocating new ones.
        let n = ALLOCATABLE.len() + 3;
        let mut instrs: Vec<Instr> = Vec::new();
        for batch in 0..2u32 {
            let base = batch * 100;
            for i in 0..n {
                instrs.push(Instr::Opaque {
                    dst: v(base + i as u32),
                    tag: "t",
                    width: Width::W64,
                    at: None,
                });
            }
            for i in 0..n {
                instrs.push(Instr::Store {
                    disp: 0,
                    addr: Operand::Imm(0x1000),
                    value: Operand::Val(v(base + i as u32)),
                    width: Width::W64,
                });
            }
        }
        let b = blk(instrs, Vec::new());
        let a = allocate(&b);
        assert!(
            a.slots <= 6,
            "slots should be reused across disjoint pressure peaks, got {}",
            a.slots
        );
        assert!(verify(&b, &a).is_empty(), "{:?}", verify(&b, &a));
    }

    /// Run a move sequence against a starting register state and report the result.
    const SPILL_BASE: u64 = 0x9000;

    fn run_moves(start: &HashMap<Reg, u64>, moves: &[Move]) -> HashMap<Reg, u64> {
        let mut st = start.clone();
        for m in moves {
            match m {
                Move::Set {
                    dst,
                    src: Loc::Reg(r),
                } => {
                    let v = *st.get(r).unwrap_or(&0);
                    st.insert(*dst, v);
                }
                // A reload is modelled as a distinct value per slot.
                Move::Set {
                    dst,
                    src: Loc::Spill(i),
                } => {
                    st.insert(*dst, SPILL_BASE + *i as u64);
                }
                Move::Imm { dst, value } => {
                    st.insert(*dst, *value);
                }
                Move::Swap { a, b } => {
                    let va = *st.get(a).unwrap_or(&0);
                    let vb = *st.get(b).unwrap_or(&0);
                    st.insert(*a, vb);
                    st.insert(*b, va);
                }
                Move::Frame { dst, offset } => {
                    // Treated as a distinct synthetic value for test purposes.
                    st.insert(*dst, SPILL_BASE.wrapping_add(*offset as u64));
                }
            }
        }
        st
    }

    /// Check that `reconcile` realizes the parallel assignment in `exits`.
    fn check_parallel(exits: Vec<(Reg, Operand)>) {
        let b = blk(Vec::new(), exits.clone());
        let a = allocate(&b);
        let moves = reconcile(&b, &a);

        // Distinct starting value per register, so any mix-up is visible.
        let start: HashMap<Reg, u64> = ALLOCATABLE
            .iter()
            .enumerate()
            .map(|(i, r)| (*r, 0x100 + i as u64))
            .collect();
        let got = run_moves(&start, &moves);

        for (dst, src) in &exits {
            let want = match src {
                Operand::Imm(v) => *v,
                Operand::Entry(r) | Operand::Param(r) => *start.get(r).expect("known reg"),
                Operand::Val(_) | Operand::Frame(_) | Operand::OutArg(_) => continue,
            };
            assert_eq!(
                got.get(dst).copied(),
                Some(want),
                "after {moves:?}, {dst:?} should hold the entry value of {src:?}"
            );
        }
    }

    #[test]
    fn a_two_cycle_is_broken_by_a_swap() {
        // rax <- rcx and rcx <- rax simultaneously. No ordering of plain moves works.
        let exits = vec![
            (Reg::Rax, Operand::Entry(Reg::Rcx)),
            (Reg::Rcx, Operand::Entry(Reg::Rax)),
        ];
        let b = blk(Vec::new(), exits.clone());
        let moves = reconcile(&b, &allocate(&b));
        assert_eq!(
            moves.len(),
            1,
            "a two-cycle needs exactly one swap: {moves:?}"
        );
        assert!(matches!(moves[0], Move::Swap { .. }));
        check_parallel(exits);
    }

    #[test]
    fn a_three_cycle_is_resolved() {
        check_parallel(vec![
            (Reg::Rax, Operand::Entry(Reg::Rcx)),
            (Reg::Rcx, Operand::Entry(Reg::Rdx)),
            (Reg::Rdx, Operand::Entry(Reg::Rax)),
        ]);
    }

    #[test]
    fn fan_out_from_one_source_is_resolved() {
        // Two destinations read the same source, and one of them is that source's own
        // destination, so order matters.
        check_parallel(vec![
            (Reg::Rbx, Operand::Entry(Reg::Rax)),
            (Reg::R12, Operand::Entry(Reg::Rax)),
            (Reg::Rax, Operand::Entry(Reg::Rbx)),
        ]);
    }

    #[test]
    fn immediates_and_cycles_mix() {
        check_parallel(vec![
            (Reg::Rax, Operand::Entry(Reg::Rcx)),
            (Reg::Rcx, Operand::Entry(Reg::Rax)),
            (Reg::Rbx, Operand::Imm(0xdead)),
        ]);
    }

    #[test]
    fn dead_definitions_do_not_share_a_register() {
        // Two values, neither read. Both must still get an interval, or they could be
        // assigned the same register and later confused for one value.
        let b = blk(
            vec![
                Instr::Opaque {
                    dst: v(0),
                    tag: "t",
                    width: Width::W64,
                    at: None,
                },
                Instr::Opaque {
                    dst: v(1),
                    tag: "t",
                    width: Width::W64,
                    at: None,
                },
            ],
            Vec::new(),
        );
        let ivs = intervals(&b);
        assert_eq!(ivs.len(), 2);
    }
    #[test]
    fn call_crossing_value_spills_only_when_non_volatiles_are_pinned() {
        // rcx is read after the call, so its live range spans it and splits.
        let instrs = || {
            vec![
                Instr::Call {
                    target: CallTarget::Direct(0x1000),
                    args: vec![],
                },
                Instr::Store {
                    addr: Operand::Entry(Reg::Rcx),
                    disp: 0,
                    value: Operand::Imm(1),
                    width: Width::W64,
                },
            ]
        };

        // Short exit list: the saved half gets a register.
        let lean = blk(instrs(), vec![(Reg::Rax, Operand::Entry(Reg::Rax))]);
        let a = allocate(&lean);
        let saved = a
            .segments
            .iter()
            .find(|s| s.item == Item::Saved(Reg::Rcx))
            .expect("rcx is read after the call, so its range splits");
        assert!(
            matches!(saved.loc, Loc::Reg(r) if !VOLATILE.contains(&r)),
            "expected a non-volatile register, got {:?}",
            saved.loc
        );

        // Every non-volatile pinned by an identity exit: nothing left to place it in.
        let pinned: Vec<(Reg, Operand)> = non_volatile()
            .into_iter()
            .map(|r| (r, Operand::Entry(r)))
            .collect();
        let crowded = blk(instrs(), pinned);
        let a2 = allocate(&crowded);
        let saved2 = a2
            .segments
            .iter()
            .find(|s| s.item == Item::Saved(Reg::Rcx))
            .expect("same split");
        assert!(
            matches!(saved2.loc, Loc::Spill(_)),
            "with all non-volatiles pinned the value must spill, got {:?}",
            saved2.loc
        );
    }

    /// A fused comparison reads its operands at the branch, so an entry value the
    /// comparison names must stay live one past the last instruction.
    #[test]
    fn a_fused_comparison_keeps_its_entry_operand_live() {
        // rcx is read by nothing in the block; only the fused comparison uses it.
        let mut b = blk(
            vec![Instr::Cast {
                dst: v(0),
                kind: CastKind::Zext,
                a: Operand::Entry(Reg::Rax),
                from: Width::W8,
                to: Width::W64,
            }],
            Vec::new(),
        );
        b.fused_cmp = Some(crate::ir::sched::FusedCmp {
            a: Operand::Entry(Reg::Rcx),
            b: Operand::Imm(0),
            op: crate::ir::expr::BinOp::Eq,
            width: Width::W64,
        });

        let rs = ranges(&b);
        let rcx = rs
            .iter()
            .find(|r| r.item == Item::Entry(Reg::Rcx))
            .expect("the fused comparison's entry operand needs a range");
        assert_eq!(
            rcx.last_use,
            b.instrs.len() + 1,
            "a fused comparison operand is read by the branch, one past the last \
             instruction; a shorter range lets another value take the register"
        );
    }

    #[test]
    fn value_live_across_a_boxed_write_avoids_its_registers() {
        let b = blk(
            vec![
                Instr::Cast {
                    dst: v(0),
                    kind: CastKind::Zext,
                    a: Operand::Entry(Reg::Rsi),
                    from: Width::W8,
                    to: Width::W64,
                },
                // `cpuid` = 0f a2, writing RAX/RBX/RCX/RDX.
                Instr::Boxed {
                    site: 0x2000,
                    text: "cpuid".into(),
                    bytes: vec![0x0f, 0xa2],
                    mem: None,
                    uses: Vec::new(),
                },
                Instr::Store {
                    disp: 0,
                    addr: Operand::Val(v(0)),
                    value: Operand::Imm(0),
                    width: Width::W64,
                },
            ],
            Vec::new(),
        );
        let ivs = intervals(&b);
        // The constraint is recorded in `avoid`, not `crosses_call`. Deliberately
        // narrower: `cpuid` rules out the four registers it writes, whereas
        // `crosses_call` would restrict the range to the eight non-volatile registers
        // and spill far more than necessary.
        assert!(
            !ivs[0].avoid.is_empty(),
            "a value read after a boxed instruction that clobbers registers must record \
             those registers, or it can be placed in one the instruction destroys"
        );
        for clobbered in [Reg::Rax, Reg::Rbx, Reg::Rcx, Reg::Rdx] {
            assert!(
                ivs[0].avoid.contains(clobbered),
                "cpuid writes {clobbered:?}, so it must be in the avoid set"
            );
        }
        let a = allocate(&b);
        let loc = a.loc_at(Item::Val(v(0)), 2);
        for clobbered in [Reg::Rax, Reg::Rbx, Reg::Rcx, Reg::Rdx] {
            assert_ne!(
                loc,
                Some(Loc::Reg(clobbered)),
                "cpuid destroys {clobbered:?}; a value live across it cannot live there"
            );
        }
        // And the verifier must agree, so a regression is caught even where the
        // allocator happens to pick a safe register by luck.
        assert!(
            verify(&b, &a).is_empty(),
            "the allocation must verify: {:?}",
            verify(&b, &a)
        );
    }
}
