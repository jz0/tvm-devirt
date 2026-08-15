//! The recovered control-flow graph: the IR every later stage consumes. 
//! Separated from the recovery engine because the two have different audiences. 
//! [`crate::vm::explore`] is the only thing that builds a [`Cfg`]; scheduling, register allocation, lowering and writeback only read one.

use crate::ir::expr::{self, Arena, Ref};
use crate::ir::lift::Event;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId {
    pub handler: u64,
    pub vip: Option<u64>,
}

impl std::fmt::Display for BlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.vip {
            Some(v) => write!(f, "{:#x}@vip{:#x}", self.handler, v),
            None => write!(f, "{:#x}", self.handler),
        }
    }
}

/// How a recovered block ends.
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Unconditional continuation to another virtual block.
    Jump(BlockId),
    /// Two-way branch recovered by pinning a flag bit.
    Branch {
        /// The guest predicate, in the owning [`Cfg`]'s arena.
        predicate: Ref,
        taken: BlockId,
        not_taken: BlockId,
    },
    /// Multi-way branch recovered by enumerating a narrow index.
    Switch {
        selector: Ref,
        targets: Vec<BlockId>,
    },
    /// The virtualized function returned to a symbolic address.
    Return { dest: Ref },
    /// Control left the function to a concrete address.
    TailCall { target: u64 },
    /// Recovery failed here.
    Unresolved { reason: String },
    /// Loop back edge to an already-recovered block.
    Backedge { target: BlockId },
    /// A two-way branch whose edges were established by the concrete pass but
    /// whose predicate could not be re-derived once guest state was symbolized.
    /// The edges are trustworthy; the condition needs recovering from the block's
    /// flag computation.
    Adopted { taken: BlockId, not_taken: BlockId },
    /// As [`Terminator::Adopted`], for a multi-way branch.
    AdoptedSwitch { targets: Vec<BlockId> },
    /// The concrete pass proved this block returns, but its destination expression
    /// could not be carried across.
    AdoptedReturn,
}

/// An SSA phi: which value a block parameter takes along each incoming edge.
#[derive(Debug, Clone)]
pub struct Phi {
    pub block: BlockId,
    pub reg: crate::ir::expr::Reg,
    /// The parameter node itself, as referenced by the block's expressions.
    pub param: Ref,
    /// One entry per predecessor. `None` means that predecessor was never
    /// recovered, so it supplied no value.
    pub incoming: Vec<(BlockId, Option<Ref>)>,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub id: BlockId,
    /// This block's index, as referenced by its [`Op::Param`]s.
    pub block_ref: expr::BlockRef,
    /// SSA parameters: the value each guest register held on entry. Every expression in this block is written in terms of these rather than in terms of function entry, so the block means the same thing no matter which predecessor reached it.
    pub params: Vec<(crate::ir::expr::Reg, Ref)>,
    /// Guest effects observed while evaluating this block, in program order.
    /// All `Ref`s point into the owning [`Cfg`]'s arena.
    pub events: Vec<Event>,
    /// Whether `exit_regs` came from a VM context (virtualized) rather than from
    /// the machine registers directly (mutation-only).
    pub from_vm_context: bool,

    /// Guest register values on exit from this block, in the owning [`Cfg`]'s
    /// arena. This is what lowering needs in order to know which value each
    /// architectural register holds at the end of the block; the events alone
    /// only describe memory and calls.
    pub exit_regs: Vec<(crate::ir::expr::Reg, Ref)>,
    pub terminator: Terminator,
    /// VM instructions interpreted to recover this block.
    pub cost: usize,
    /// Predecessors, filled in after recovery.
    pub preds: Vec<BlockId>,
}

#[derive(Debug)]
pub struct Cfg {
    pub entry: BlockId,
    /// Owns every expression referenced by the blocks. Recovery forks the
    /// evaluator per path, so each block's values are born in a short-lived
    /// arena; they are grafted here so the recovered CFG is self-contained.
    pub arena: Arena,
    pub blocks: Vec<Block>,
    pub total_steps: usize,
    pub unresolved: usize,
    /// Edges that close a loop.
    #[allow(dead_code)] // retained for the optional batch diagnostics command
    pub back_edges: Vec<(BlockId, BlockId)>,
    /// Block parameters removed because their phi was trivial.
    pub trivial_phis_removed: usize,
    /// Set when the wall-clock timer rather than the work budget ended exploration.
    /// The result then depends on machine load and is not reproducible.
    pub timed_out: bool,
    /// The concrete value RSP was pinned to during recovery. Every stack reference in
    /// the recovered code is an absolute address derived from it, and lowering must
    /// turn those back into RSP-relative form, so the anchor has to travel with the CFG.
    pub stack_base: u64,
    /// Half-open VA range of the section holding VM bytecode. Validation needs it to tell a VM handler from a native address. Handlers are shared by construction; one dispatcher serves every branch in a function; so a handler without a bytecode position is ambiguous.
    pub vm_range: (u64, u64),
}

impl Cfg {
    pub fn block(&self, id: BlockId) -> Option<&Block> {
        self.blocks.iter().find(|b| b.id == id)
    }

    /// Incoming values for every block parameter, i.e. the PHI nodes. A block's parameter for register `R` takes, along each incoming edge, the value that predecessor's `exit_regs` gave `R`.
    pub fn phis(&self) -> Vec<Phi> {
        let exit_of: HashMap<BlockId, &Vec<(crate::ir::expr::Reg, Ref)>> =
            self.blocks.iter().map(|b| (b.id, &b.exit_regs)).collect();

        let mut out = Vec::new();
        for b in &self.blocks {
            for (reg, param) in &b.params {
                let mut incoming = Vec::new();
                for p in &b.preds {
                    // A predecessor that was never recovered (an unresolved or
                    // budget-truncated block) contributes no value. Recording the
                    // edge as `None` keeps it visible rather than silently
                    // narrowing the PHI to the paths that happened to work.
                    let v = exit_of
                        .get(p)
                        .and_then(|regs| regs.iter().find(|(r, _)| r == reg))
                        .map(|(_, v)| *v);
                    incoming.push((*p, v));
                }
                out.push(Phi {
                    block: b.id,
                    reg: *reg,
                    param: *param,
                    incoming,
                });
            }
        }
        out
    }

    /// Eliminate trivial phis, returning how many were removed. A phi is trivial when, ignoring edges that carry the parameter itself, all incoming values are the same: the parameter then *is* that value.
    pub fn simplify_phis(&mut self) -> usize {
        let mut removed = 0usize;
        loop {
            let mut map: HashMap<Ref, Ref> = HashMap::new();
            for phi in self.phis() {
                let mut vals: Vec<Ref> = phi
                    .incoming
                    .iter()
                    .filter_map(|(_, v)| *v)
                    .filter(|v| *v != phi.param)
                    .collect();
                vals.sort();
                vals.dedup();
                // Exactly one distinct non-self value. A phi with no incoming
                // value at all is left alone: that means every predecessor failed
                // to recover, and inventing a value would be a lie.
                if let [only] = vals[..] {
                    if only != phi.param {
                        map.insert(phi.param, only);
                    }
                }
            }
            if map.is_empty() {
                return removed;
            }
            removed += map.len();
            self.apply_rewrite(&map);
        }
    }

    /// Rewrite every expression the CFG holds through `map`.
    fn apply_rewrite(&mut self, map: &HashMap<Ref, Ref>) {
        let arena = &mut self.arena;
        for b in &mut self.blocks {
            for e in &mut b.events {
                rewrite_event(arena, e, map);
            }
            for (_, v) in &mut b.exit_regs {
                *v = arena.rewrite(*v, map);
            }
            b.params.retain(|(_, p)| !map.contains_key(p));
            match &mut b.terminator {
                Terminator::Branch { predicate, .. } => {
                    *predicate = arena.rewrite(*predicate, map);
                }
                Terminator::Switch { selector, .. } => {
                    *selector = arena.rewrite(*selector, map);
                }
                Terminator::Return { dest } => *dest = arena.rewrite(*dest, map),
                _ => {}
            }
        }
    }

    /// Remove parameters that the block's body never reads.
    pub fn remove_dead_params(&mut self) {
        for b in &mut self.blocks {
            let mut refs: Vec<Ref> = Vec::new();
            for (_, v) in &b.exit_regs {
                refs.push(*v);
            }
            for e in &b.events {
                match e {
                    Event::Store { addr, value, .. } | Event::Load { addr, value, .. } => {
                        refs.push(*addr);
                        refs.push(*value);
                    }
                    Event::Call { target, args, .. } => {
                        if let Some(t) = target {
                            refs.push(*t);
                        }
                        refs.extend(args.iter().map(|(_, v)| *v));
                    }
                    Event::Boxed { .. } => {}
                }
            }
            match &b.terminator {
                Terminator::Branch { predicate, .. } => refs.push(*predicate),
                Terminator::Switch { selector, .. } => refs.push(*selector),
                Terminator::Return { dest } => refs.push(*dest),
                _ => {}
            }
            let mut used: HashSet<Ref> = HashSet::new();
            for r in refs {
                for leaf in expr::leaves(&self.arena, r) {
                    used.insert(leaf);
                }
            }
            b.params.retain(|(_, p)| used.contains(p));
        }
    }

    pub fn successors(b: &Block) -> Vec<BlockId> {
        match &b.terminator {
            Terminator::Jump(t) => vec![*t],
            Terminator::Branch {
                taken, not_taken, ..
            } => vec![*taken, *not_taken],
            Terminator::Switch { targets, .. } => targets.clone(),
            Terminator::Backedge { target } => vec![*target],
            Terminator::Adopted { taken, not_taken } => vec![*taken, *not_taken],
            Terminator::AdoptedSwitch { targets } => targets.clone(),
            _ => Vec::new(),
        }
    }
}

pub fn terminator_kind(t: &Terminator) -> &'static str {
    match t {
        Terminator::Jump(_) => "jump",
        Terminator::Branch { .. } => "branch",
        Terminator::Switch { .. } => "switch",
        Terminator::Return { .. } => "return",
        Terminator::TailCall { .. } => "tailcall",
        Terminator::Unresolved { .. } => "unresolved",
        Terminator::Backedge { .. } => "backedge",
        Terminator::Adopted { .. } => "branch (edges only)",
        Terminator::AdoptedSwitch { .. } => "switch (edges only)",
        Terminator::AdoptedReturn => "return (adopted)",
    }
}

pub fn summarize(cfg: &Cfg) -> HashMap<&'static str, usize> {
    let mut m = HashMap::new();
    for b in &cfg.blocks {
        *m.entry(terminator_kind(&b.terminator)).or_insert(0) += 1;
    }
    m
}

/// Rewrite the expressions inside an event through `map`.
fn rewrite_event(arena: &mut Arena, e: &mut Event, map: &HashMap<Ref, Ref>) {
    match e {
        Event::Store { addr, value, .. } => {
            *addr = arena.rewrite(*addr, map);
            *value = arena.rewrite(*value, map);
        }
        Event::Load { addr, value, .. } => {
            *addr = arena.rewrite(*addr, map);
            *value = arena.rewrite(*value, map);
        }
        // `rsp` is a concrete stack pointer, not an expression, so there is
        // nothing in it to rewrite.
        Event::Call { target, args, .. } => {
            if let Some(t) = target {
                *t = arena.rewrite(*t, map);
            }
            // Arguments are ordinary expressions and take part in SSA renaming
            // like any other operand.
            for (_, v) in args.iter_mut() {
                *v = arena.rewrite(*v, map);
            }
        }
        Event::Boxed { .. } => {}
    }
}
