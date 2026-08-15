//! x86 semantics over the abstract state ("guided symbolic evaluation"). single-step the VM's own code with RSP concretized,
//! let constant folding collapse the MBA-obfuscated dispatch arithmetic, and follow the resulting computed jumps.

use crate::binary::disasm;
use crate::binary::pe::PeFile;
use crate::ir::expr::{Arena, BinOp, Op, Ref, Reg, UnOp, Width};
use crate::ir::hash;
use crate::vm::state::{Flag, State};
use iced_x86::{
    Instruction, InstructionInfoFactory, Mnemonic, OpAccess, OpCodeOperandKind, OpKind, Register,
};
use std::collections::{HashMap, HashSet};

/// Order of the guest register image in the VM context.
pub const GUEST_IMAGE_ORDER: [Reg; 16] = [
    Reg::Rax,
    Reg::Rbx,
    Reg::Rcx,
    Reg::Rdx,
    Reg::Rsp,
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

const GUEST_IMAGE_ORDER_ALT: [Reg; 16] = [
    Reg::Rax,
    Reg::Rbp,
    Reg::Rbx,
    Reg::Rcx,
    Reg::Rdi,
    Reg::Rdx,
    Reg::Rsi,
    Reg::Rsp,
    Reg::R8,
    Reg::R9,
    Reg::R10,
    Reg::R11,
    Reg::R12,
    Reg::R13,
    Reg::R14,
    Reg::R15,
];

/// A candidate guest-register-image layout: the slot order and where the packed
/// flags word sits relative to the registers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Layout {
    pub order: &'static [Reg; 16],
    /// Bytes from the image base to register slot 0. Non-zero when the flags word
    /// leads the image.
    pub first_reg: u64,
}

/// Layouts to try, in the order they are preferred when scores tie.
pub const LAYOUTS: [Layout; 2] = [
    Layout {
        order: &GUEST_IMAGE_ORDER,
        first_reg: 0,
    },
    Layout {
        order: &GUEST_IMAGE_ORDER_ALT,
        first_reg: 8,
    },
];

impl Layout {
    /// Address of `reg`'s slot in an image based at `base`.
    pub fn slot(&self, base: u64, reg: Reg) -> Option<u64> {
        let i = self.order.iter().position(|r| *r == reg)?;
        Some(base.wrapping_add(self.first_reg + i as u64 * 8))
    }
}

/// How far from `stack_base` a guest RSP is still considered plausible when
/// identifying the guest register image.
const GUEST_FRAME_WINDOW: u64 = 0x2000;

/// Imports that never return to their caller. Kept as an exact-name allowlist to
/// avoid truncating calls to similarly named routines that do return.
const NORETURN_IMPORTS: [&str; 5] = [
    "KeBugCheck",
    "KeBugCheckEx",
    "ExRaiseStatus",
    "ExRaiseAccessViolation",
    "ExRaiseDatatypeMisalignment",
];

/// IAT slots in `pe` whose import never returns. Resolved once when the emulator is built, because the check runs at every call site and walking the import descriptors each time would be wasteful.
fn noreturn_import_slots(pe: &PeFile) -> Vec<u64> {
    pe.imports()
        .into_iter()
        .filter(|(_, _, sym)| NORETURN_IMPORTS.contains(&sym.as_str()))
        .map(|(slot, _, _)| slot)
        .collect()
}

/// How far [`callee_never_returns`] scans before giving up. A throw wrapper is a handful of instructions: set up one argument, call, trap.
const NORETURN_SCAN_INSTRS: usize = 32;

pub const ARG_REGS: [Reg; 4] = [Reg::Rcx, Reg::Rdx, Reg::R8, Reg::R9];
/// Minimum number of registers that must still hold their entry values for a
/// candidate guest register image to be accepted.
const MIN_CONTEXT_SCORE: usize = 2;

/// How many guest registers must be seen spilled to consecutive context slots before the implied base is believed.
const MIN_SPILL_REGS: usize = 6;

/// Size of the guest register image: a packed-flags word plus 16 registers.
const GUEST_IMAGE_BYTES: u64 = 17 * 8;

/// Which part of the machine a memory access touches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// Guest-visible memory: the guest's own stack frame, globals, heap.
    Guest,
    /// The VM's private area below the guest stack: register file, spills.
    VmScratch,
    /// Bytecode stream, dispatch tables, and other image reads.
    VmImage,
    /// A symbolic address not rooted in guest state: the VM indexing its own
    /// structures.
    VmInternal,
}

impl Region {
    /// Whether an access here carries guest program semantics.
    pub fn is_guest(self) -> bool {
        matches!(self, Region::Guest)
    }
}

/// Guest-visible effect observed during evaluation.
#[derive(Debug, Clone)]
pub enum Event {
    /// A store whose address is *not* inside the VM's own scratch area.
    Store {
        addr: Ref,
        value: Ref,
        width: Width,
        site: u64,
        region: Region,
    },
    /// A load from an address the evaluator could not resolve statically.
    Load {
        addr: Ref,
        value: Ref,
        width: Width,
        site: u64,
        region: Region,
    },
    /// An instruction executed natively by the VM (a "boxed" instruction). `bytes` is the original encoding, so codegen can re-decode and re-encode it rather than trying to reassemble from `text`.
    Boxed {
        site: u64,
        text: String,
        bytes: Vec<u8>,
        mem: Option<BoxedMem>,
        /// The architectural registers this instruction leaves a value in, paired with the opaque naming that value.
        defs: Vec<(Register, Ref)>,
        /// The architectural registers this instruction *reads*, paired with the guest value each held. The mirror image of `defs`, and needed for the same reason.
        uses: Vec<(Register, Ref)>,
    },
    /// A call out of the virtualized function.
    Call {
        target: Option<Ref>,
        site: u64,
        rsp: Option<u64>,
        /// Address of the import table slot, when the call goes through one.
        import_slot: Option<u64>,
        /// What the Win64 argument registers held *before* the call clobbered
        /// them. Without this the argument setup is lost: `clobber_call`
        /// overwrites RCX/RDX/R8/R9 with opaques, so a later reader sees only
        /// the callee's return convention and no arguments at all.
        args: Vec<(Reg, Ref)>,
        ret: Option<Ref>,
    },
}

/// The memory operand of a boxed instruction.
#[derive(Debug, Clone)]
pub struct BoxedMem {
    /// Symbolic address of the operand.
    pub addr: Ref,
    /// Size of the access in bytes. SSE operands are 16, which no `Width` covers,
    /// so this is a raw byte count rather than a `Width`.
    pub bytes: u32,
    /// Whether the instruction writes the operand (operand 0 is memory).
    pub writes: bool,
}

/// Why evaluation stopped.
#[derive(Debug, Clone)]
pub enum Stop {
    /// Indirect branch whose destination stayed symbolic. Either a virtualized
    /// JCC or incomplete folding.
    SymbolicBranch { site: u64, dest: Ref },
    /// A native `JCC` on a guest predicate. Both targets are statically known;
    /// only the predicate is symbolic.
    NativeBranch {
        site: u64,
        predicate: Ref,
        taken: u64,
        not_taken: u64,
    },
    /// `ret` with a concrete/symbolic destination: the function returned.
    Return { site: u64, dest: Ref },
    /// Instruction the lifter does not model.
    Unsupported { site: u64, text: String },
    /// Could not read code at the given address.
    Unreadable { site: u64 },
    /// A dispatch destination that repeats closes a loop.
    Backedge {
        site: u64,
        target: u64,
        vip: Option<u64>,
    },
    /// Step budget exhausted.
    Budget { site: u64 },
    /// The expression DAG grew past its limit, which means folding has broken
    /// down and further evaluation would only produce garbage.
    Diverged { site: u64, nodes: usize },
    /// Reached an address outside the image.
    OutOfImage { site: u64 },
    /// A call to a routine that never returns. The block ends at the call.
    NoReturn { site: u64 },
}

/// Result of a single step.
pub enum Step {
    Next(u64),
    /// Conditional branch whose predicate folded; already resolved.
    Stopped(Stop),
}

/// How many recently interpreted instruction addresses an [`Emulator`] retains.
///
/// Sized well above any useful `inspect --tail N` so the flag behaves as if the whole
/// history were kept, while bounding what a forked path has to copy.
const TRACE_TAIL_CAP: usize = 4096;

#[derive(Clone)]
pub struct Emulator<'a> {
    pub pe: &'a PeFile,
    pub arena: Arena,
    pub state: State,
    pub events: Vec<Event>,
    /// Base value RSP was concretized to.
    pub stack_base: u64,
    /// Instruction addresses visited, most recent last, capped at [`TRACE_TAIL_CAP`] entries. Recovery only ever reads the *count* of instructions interpreted, to charge a block its cost; the addresses themselves are a debugging aid for `inspect --tail`.
    trace: std::collections::VecDeque<u64>,
    /// Total instructions interpreted, counting those dropped from `trace`.
    pub steps: usize,
    /// Every resolved indirect branch: (site, destination). These are the VM
    /// handler transitions, i.e. the bytecode program being decoded.
    pub dispatches: Vec<(u64, u64)>,
    /// Loads from a concrete address with no image backing: (site, addr, width).
    /// A load here that the VM expected to resolve means an earlier fold is
    /// wrong, so this is the first place to look when dispatch gets stuck.
    pub unmapped_loads: Vec<(u64, u64, Width)>,
    /// Loads whose address never folded to a constant: (site, width). These are
    /// the precision losses that eventually stall dispatch.
    pub symbolic_loads: Vec<(u64, Width)>,
    /// Leaves pinned to concrete values for this run. Used to resolve
    /// virtualized conditional branches by exploring each side separately: an
    /// unpinned flag bit keeps the dispatch arithmetic from folding at all.
    pub pins: Vec<(Ref, u64)>,
    /// Diagnostic: why the last context search failed, as
    /// (found a plausible RSP slot, best InitReg score seen).
    pub ctx_miss: Option<(bool, usize)>,
    /// Cached candidate base of the guest register image. Later searches may replace
    /// it when the VM relocates the live image.
    pub guest_ctx: Option<u64>,
    /// Whether reaching an already-recovered block entry stops evaluation.
    pub stop_on_revisit: bool,
    /// Identities of already-recovered blocks, as (handler, VIP) pairs. Reaching one again is a loop back edge. The VIP component is essential: Tencent VM shares handlers, so the same `.tvm0` address is entered many times at different points in the bytecode program.
    block_entries: std::collections::HashSet<(u64, Option<u64>)>,
    /// Number of guest events recorded when the current block began running. A back edge is only real if the cycle through the block did something observable.
    events_at_block_entry: usize,
    /// Value stored at `[rbp+0]`: the phantom-unwind dummy function pointer the
    /// VM keeps there for SEH. It is a `.tvm0` pointer, so VIP detection has to
    /// filter it out explicitly.
    unwind_slot: Option<u64>,
    /// Persistent memo for [`Self::substitute`].
    subst_memo: std::collections::HashMap<Ref, Ref>,
    /// Number of pins the memo was built against; a change invalidates it.
    memo_pin_count: usize,
    /// Cache for [`Self::depends_on_pins`].
    pin_dep: std::collections::HashMap<Ref, bool>,
    /// Upper bound on expression DAG size before evaluation is abandoned.
    pub node_limit: usize,
    /// Treat instructions outside the modelled AMD64 subset as boxed
    /// instructions (record and continue) rather than stopping.
    pub box_unmodelled: bool,
    /// Treat CET queries (`rdsspq`) as "CET disabled". Safe because the shadow
    /// stack has no guest semantics: the VM only uses it to keep its own
    /// non-returning CALLs from overflowing the shadow stack.
    pub assume_cet_disabled: bool,
    /// `[lo, hi)` VA range of the VM section. Calls landing inside it are VM
    /// control flow to be inlined; calls leaving it are guest semantics.
    pub vm_range: (u64, u64),
    /// IAT slots of imports that never return, so a call through one ends the
    /// block instead of falling through to whatever byte follows.
    noreturn_slots: Vec<u64>,
    /// Every IAT slot in the image. An exact set rather than a range test, because `jmp qword ptr [addr]` is also how the VM dispatches to its own handlers.
    import_slots: std::collections::HashSet<u64>,
    /// Argument registers an instruction has written since the last call. This is what separates a real argument from a leftover, and it has to be a record of *writes*, not a test on the value.
    written_args: std::collections::HashSet<Reg>,
    /// Layout the guest register image was found to use, once known. Cached with
    /// the base, since the two are only meaningful together.
    pub guest_layout: Option<Layout>,
    pub vip_slot: Option<u64>,
}

impl<'a> Emulator<'a> {
    pub fn new(pe: &'a PeFile, stack_base: u64) -> Self {
        Self::with_vm_section(pe, stack_base, ".tvm0")
    }

    pub fn with_vm_section(pe: &'a PeFile, stack_base: u64, vm_section: &str) -> Self {
        let mut arena = Arena::new();
        let state = State::new(&mut arena, stack_base);

        let vm_range = pe
            .section_by_name(vm_section)
            .map(|s| {
                let lo = pe.rva_to_va(s.virtual_address);
                (lo, lo + s.virtual_size.max(s.raw_size) as u64)
            })
            .unwrap_or((0, 0));
        Self {
            pe,
            arena,
            state,
            events: Vec::new(),
            stack_base,
            trace: std::collections::VecDeque::new(),
            steps: 0,
            dispatches: Vec::new(),
            unmapped_loads: Vec::new(),
            symbolic_loads: Vec::new(),
            pins: Vec::new(),
            ctx_miss: None,
            guest_ctx: None,
            stop_on_revisit: true,
            block_entries: std::collections::HashSet::new(),
            events_at_block_entry: 0,
            unwind_slot: None,
            subst_memo: std::collections::HashMap::new(),
            memo_pin_count: 0,
            pin_dep: std::collections::HashMap::new(),
            node_limit: std::env::var("TVM_NODE_LIMIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1_000_000),
            box_unmodelled: true,
            assume_cet_disabled: true,
            vm_range,
            written_args: std::collections::HashSet::new(),
            guest_layout: None,
            // Set by the caller from `discover_vip_slot` before recovery proper.
            // Left `None` here so that nothing silently depends on a default offset:
            // a wrong constant would degenerate `(handler, vip)` to `handler` and
            // close back edges that do not exist.
            vip_slot: None,
            noreturn_slots: noreturn_import_slots(pe),
            import_slots: pe.imports().into_iter().map(|(slot, _, _)| slot).collect(),
        }
    }

    pub fn in_vm(&self, va: u64) -> bool {
        (self.vm_range.0..self.vm_range.1).contains(&va)
    }

    /// Read a fully-constant qword out of the abstract memory.
    fn read_const_qword(&self, addr: u64) -> Option<u64> {
        let mut bytes = [0u8; 8];
        for i in 0..8u64 {
            match self.state.mem.get(&addr.wrapping_add(i)) {
                Some(crate::vm::state::Byte::Const(c)) => bytes[i as usize] = *c,
                _ => return None,
            }
        }
        Some(u64::from_le_bytes(bytes))
    }

    fn pinned_slot_value(&self, addr: u64) -> Option<u64> {
        let crate::vm::state::Byte::Sym { expr, index: 0 } = self.state.mem.get(&addr)? else {
            return None;
        };
        if !matches!(self.arena.op(*expr), Op::Param(..)) {
            return None;
        }
        self.pins.iter().find(|(r, _)| r == expr).map(|(_, c)| *c)
    }

    pub fn current_vip(&self) -> Option<u64> {
        self.vip_at_depth(Self::VIP_DEPTH)
    }

    /// Bound on the VIP expression hash.
    pub const VIP_DEPTH: u32 = 12;

    fn vip_at_depth(&self, vdepth: u32) -> Option<u64> {
        let base = self.arena.as_const(self.state.reg(Reg::Rbp))?;
        let vip_off = self.vip_slot?;
        let mut h = hash::FNV1A_OFFSET_BASIS;
        let mix = |off: u64, v: u64, h: &mut u64| {
            // FNV-1a over (offset, value): a pointer moving to another slot is a
            // different position, so the offset has to be mixed in.
            hash::mix_u64(off, h);
            hash::mix_u64(v, h);
        };

        match self.read_const_qword(base.wrapping_add(vip_off)) {
            Some(vip) => {
                if !self.in_vm(vip) || Some(vip) == self.unwind_slot {
                    return None;
                }
                mix(vip_off, vip, &mut h);
            }
            None => {
                let sym = self.slot_expr(base.wrapping_add(vip_off))?;
                mix(vip_off, self.arena.structural_hash(sym, vdepth), &mut h);
            }
        }
        for off in (vip_off + 8..vip_off + 8 + 0xa8).step_by(8) {
            if let Some(v) = self.read_const_qword(base.wrapping_add(off)) {
                if self.in_vm(v) && Some(v) != self.unwind_slot {
                    mix(off, v, &mut h);
                }
            }
        }
        Some(h)
    }

    /// The expression stored in a fully-symbolic qword slot, if all eight bytes come from one value at their natural positions.
    fn slot_expr(&self, addr: u64) -> Option<Ref> {
        let mut found: Option<Ref> = None;
        for i in 0..8u32 {
            match self.state.mem.get(&addr.wrapping_add(i as u64)) {
                Some(crate::vm::state::Byte::Sym { expr, index }) if *index == i => match found {
                    Some(e) if e != *expr => return None,
                    _ => found = Some(*expr),
                },
                _ => return None,
            }
        }
        found
    }

    /// VIP identifying the block that begins at `handler`.
    pub fn vip_at_entry(&self, handler: u64) -> Option<u64> {
        let _ = handler;
        self.current_vip()
    }

    /// Read a context slot as an expression, whether concrete or symbolic.
    /// Read a qword slot without assuming any layout. For diagnosing the guest
    /// register image against known-good output.
    pub fn raw_slot(&mut self, addr: u64) -> Option<Ref> {
        self.read_slot(addr)
    }

    fn read_slot(&mut self, addr: u64) -> Option<Ref> {
        let image = |a: u64| self.pe.image_u8(a);
        self.state
            .load_concrete(&mut self.arena, addr, Width::W64, image)
    }

    /// Guest registers, read from the cached context base, locating it first if necessary. `None` means this function has no VM context.
    pub fn guest_registers(&mut self) -> Option<Vec<(Reg, Ref)>> {
        let base = self.locate_guest_context()?;
        let (_, _, regs) = self.probe_context_at(base);
        if regs.is_empty() { None } else { Some(regs) }
    }

    pub fn locate_guest_context(&mut self) -> Option<u64> {
        if let Some(cached) = self.guest_ctx {
            let (cached_score, _, _) = self.probe_context_at(cached);
            // The speculative search records a miss when it fails, but falling back to
            // the cached base is a success, so the miss must not be left behind.
            let saved_miss = self.ctx_miss;
            let found = self.search_guest_context();
            self.ctx_miss = saved_miss;
            if let Some(fresh) = found {
                let (fresh_score, _, _) = self.probe_context_at(fresh);
                if fresh != cached && fresh_score > cached_score {
                    self.guest_ctx = Some(fresh);
                    let (_, _, _, layout) = self.probe_best_layout(fresh);
                    self.guest_layout = Some(layout);
                    return Some(fresh);
                }
            }
            return Some(cached);
        }
        // Two independent locators. The entry-spill pattern is tried first because it observes the context actually being *built*, which does not depend on where RBP happens to point when evaluation stops.
        let base = self
            .context_from_entry_spills()
            .or_else(|| self.search_guest_context());
        self.guest_ctx = base;
        if let Some(b) = base {
            let (_, _, _, layout) = self.probe_best_layout(b);
            self.guest_layout = Some(layout);
        }
        if base.is_some() {
            // Events recorded before the context was known were classified with no VM region to compare against, so the whole entry spill sequence was tagged as guest.
            self.reclassify_events();
        }
        base
    }

    /// Re-place already-recorded events now that the VM context is known.
    fn reclassify_events(&mut self) {
        let mut fixed = std::mem::take(&mut self.events);
        for e in &mut fixed {
            let (addr, region) = match e {
                Event::Store { addr, region, .. } | Event::Load { addr, region, .. } => {
                    (*addr, region)
                }
                _ => continue,
            };
            // Only concrete addresses are re-placed. A symbolic one was classified
            // by provenance, which does not depend on the context base.
            if let Some(a) = self.arena.as_const(addr) {
                *region = self.region_of_addr(a);
            }
        }
        self.events = fixed;
    }

    /// Locate the guest register image from the entry sequence that fills it. On entry the VM spills every guest register into the context, so the store log contains `Store { addr: k, value: InitReg(r) }` for each register at its own fixed slot.
    fn context_from_entry_spills(&self) -> Option<u64> {
        let mut votes: HashMap<u64, HashSet<Reg>> = HashMap::new();
        for e in &self.events {
            let Event::Store {
                addr,
                value,
                width: Width::W64,
                ..
            } = e
            else {
                continue;
            };
            let (Some(a), Op::InitReg(reg)) = (self.arena.as_const(*addr), self.arena.op(*value))
            else {
                continue;
            };
            // Vote under every candidate layout: each implies a different base for
            // the same store, and the layout that is actually in use is the one whose
            // implied base collects votes from many registers at once.
            for layout in LAYOUTS {
                let Some(slot) = layout.order.iter().position(|r| r == reg) else {
                    continue;
                };
                let Some(implied) = a.checked_sub(layout.first_reg + slot as u64 * 8) else {
                    continue;
                };
                votes.entry(implied).or_default().insert(*reg);
            }
        }
        votes
            .into_iter()
            .filter(|(_, regs)| regs.len() >= MIN_SPILL_REGS)
            .max_by_key(|(base, regs)| (regs.len(), *base))
            .map(|(base, _)| base)
    }

    /// Locate the guest register image inside the VM context by content.
    fn search_guest_context(&mut self) -> Option<u64> {
        // A symbolic RBP means there is no frame to search around. This is normal
        // at a guest return, where the VM has already restored guest RBP.
        let Some(rbp) = self.arena.as_const(self.state.reg(Reg::Rbp)) else {
            self.ctx_miss = Some((false, 0));
            return None;
        };
        let lo = rbp.saturating_sub(0x40);
        let hi = rbp.saturating_add(0x200);

        let mut best: Option<(usize, u64)> = None;
        let mut best_rsp_only = 0usize;
        let mut any_rsp = false;
        let mut base = lo & !7;
        while base < hi {
            let (score, rsp_ok, _) = self.probe_context_at(base);
            if rsp_ok {
                any_rsp = true;
                best_rsp_only = best_rsp_only.max(score);
            }
            if rsp_ok && score >= MIN_CONTEXT_SCORE && best.is_none_or(|(b, _)| score > b) {
                best = Some((score, base));
            }
            base += 8;
        }
        if best.is_none() {
            self.ctx_miss = Some((any_rsp, best_rsp_only));
        }
        best.map(|(_, base)| base)
    }

    /// Overwrite the guest register image with fresh SSA parameters, returning the parameter for each register. This is what makes a recovered block's expressions independent of the path that reached it.
    pub fn seed_guest_params(&mut self, block: crate::ir::expr::BlockRef) -> Vec<(Reg, Ref)> {
        let Some(base) = self.locate_guest_context() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let layout = self.guest_layout.unwrap_or(LAYOUTS[0]);
        for (i, reg) in layout.order.iter().enumerate() {
            let addr = base.wrapping_add(layout.first_reg + i as u64 * 8);
            let old = self.read_slot(addr).and_then(|v| self.arena.as_const(v));
            let p = self.arena.param(block, *reg);
            self.state.store_concrete(&self.arena, addr, p, Width::W64);
            if let Some(c) = old {
                self.pins.push((p, c));
            }
            out.push((*reg, p));
        }
        out
    }

    /// Probe a candidate guest register image at `base`, returning (InitReg matches, whether the RSP slot is plausible, the slots). Slots that cannot be read are omitted rather than failing the whole image.
    fn probe_context_at(&mut self, base: u64) -> (usize, bool, Vec<(Reg, Ref)>) {
        let (score, rsp, regs, _) = self.probe_best_layout(base);
        (score, rsp, regs)
    }

    /// Probe `base` under every candidate layout and keep the best fit.
    fn probe_best_layout(&mut self, base: u64) -> (usize, bool, Vec<(Reg, Ref)>, Layout) {
        let mut best: Option<(usize, bool, Vec<(Reg, Ref)>, Layout)> = None;
        for layout in LAYOUTS {
            let (score, rsp, regs) = self.probe_layout_at(base, layout);
            let better = match &best {
                None => true,
                Some((bs, brsp, _, _)) => (rsp, score) > (*brsp, *bs),
            };
            if better {
                best = Some((score, rsp, regs, layout));
            }
        }
        best.expect("LAYOUTS is non-empty")
    }

    fn probe_layout_at(&mut self, base: u64, layout: Layout) -> (usize, bool, Vec<(Reg, Ref)>) {
        let mut regs: Vec<(Reg, Ref)> = Vec::new();
        let mut score = 0usize;
        let mut rsp_plausible = false;

        for (i, reg) in layout.order.iter().enumerate() {
            let addr = base.wrapping_add(layout.first_reg + i as u64 * 8);
            let Some(v) = self.read_slot(addr) else {
                continue;
            };
            if slot_confirms_register(self.arena.op(v), *reg) {
                score += 1;
            }
            if *reg == Reg::Rsp {
                if let Some(c) = self.arena.as_const(v) {
                    rsp_plausible = c % 8 == 0
                        && c > self.stack_base.saturating_sub(GUEST_FRAME_WINDOW)
                        && c <= self.stack_base.saturating_add(GUEST_FRAME_WINDOW);
                }
            }
            regs.push((*reg, v));
        }

        (score, rsp_plausible, regs)
    }

    /// Register a block identity, so that reaching it again reports a back edge
    /// instead of re-lifting the loop body.
    pub fn mark_visited(&mut self, handler: u64, vip: Option<u64>) {
        self.block_entries.insert((handler, vip));
    }

    /// Note that a block is starting, for the "a loop must make progress" chk.
    pub fn begin_block(&mut self) {
        self.events_at_block_entry = self.guest_event_count();
    }

    /// Count of events that carry guest program semantics.
    fn guest_event_count(&self) -> usize {
        self.events
            .iter()
            .filter(|e| match e {
                Event::Store { region, .. } | Event::Load { region, .. } => region.is_guest(),
                Event::Boxed { .. } | Event::Call { .. } => true,
            })
            .count()
    }

    /// Clear the back-edge history.
    pub fn clear_visited(&mut self) {
        self.block_entries.clear();
    }

    /// True when `addr` is in the region the VM uses for its own context and spills, i.e. below the guest stack pointer at entry.
    fn is_vm_scratch(&self, addr: u64) -> bool {
        // With no context observed there is no VM, so nothing is VM-private. This
        // is the mutation-only case, where the machine state is the guest state
        // and the function's prologue pushes are genuine guest stores.
        if self.guest_ctx.is_none() {
            return false;
        }

        // The VM's scratch is stack memory, so an address that belongs to a
        // section of the image is never scratch no matter where it sits relative
        // to the stack.
        if self.pe.section_for_va(addr).is_some() {
            return false;
        }

        addr < self.guest_stack_floor()
    }

    /// The lowest address the guest itself can be using, i.e. its stack pointer. Read from context's RSP slot when available, since that is the guest's real stack pointer; the machine RSP belongs to the interpreter.
    fn guest_stack_floor(&self) -> u64 {
        let from_ctx = self.guest_ctx.and_then(|base| {
            let layout = self.guest_layout.unwrap_or(LAYOUTS[0]);
            let slot = layout.slot(base, Reg::Rsp)?;
            self.read_const_qword(slot)
                .or_else(|| self.pinned_slot_value(slot))
        });
        // The slot is only believed when it actually looks like a guest stack
        // pointer: at or below the entry RSP, and outside VM's own context.
        match from_ctx {
            Some(rsp)
                if rsp <= self.stack_base
                    && self
                        .guest_ctx
                        .is_some_and(|base| rsp > base + GUEST_IMAGE_BYTES) =>
            {
                rsp
            }
            _ => self.stack_base,
        }
    }

    /// Classify address that did not fold to a constant.
    fn region_of_symbolic(&self, addr: Ref) -> Region {
        if let Some((BinOp::Add, x, y)) = self.arena.as_bin(addr) {
            for (a, b) in [(x, y), (y, x)] {
                let _ = b;
                if let Some(c) = self.arena.as_const(a) {
                    if self.is_vm_scratch(c) || self.in_vm(c) {
                        return Region::VmInternal;
                    }
                }
            }
        }
        if crate::ir::expr::is_guest_rooted(&self.arena, addr) {
            Region::Guest
        } else {
            Region::VmInternal
        }
    }

    /// Which region concrete address belongs to.
    fn region_of_addr(&self, addr: u64) -> Region {
        if self.is_vm_scratch(addr) {
            return Region::VmScratch;
        }
        if self.pe.read_va(addr, 1).is_some() {
            // Backed by image data. Guest access to a global also lands here, so
            // this is only VM bookkeeping if it is reading the VM's own section.
            return if self.in_vm(addr) {
                Region::VmImage
            } else {
                Region::Guest
            };
        }
        Region::Guest
    }

    /// Read x86 register, honouring partial-register widths.
    fn read_reg(&mut self, reg: Register) -> Ref {
        if reg == Register::None {
            return self.arena.constant(0, Width::W64);
        }
        if reg == Register::RIP {
            // Handled by callers that need it; a bare RIP read is a bug.
            return self.arena.opaque("rip", Width::W64);
        }
        let Some((base, offset, width)) = decode_reg(reg) else {
            return self.arena.opaque("xreg", Width::W64);
        };
        let full = self.state.reg(base);
        if offset > 0 {
            // ah/bh/ch/dh
            let sh = self.arena.constant(offset as u64 * 8, Width::W64);
            let sh = self.arena.bin(BinOp::Shr, full, sh);
            return self.arena.trunc(sh, width);
        }
        self.arena.trunc(full, width)
    }

    /// Write x86 register with correct partial-write behaviour: 32-bit writes
    /// zero-extend, 8/16-bit writes merge into the existing value.
    fn write_reg(&mut self, reg: Register, value: Ref) {
        let Some((base, offset, width)) = decode_reg(reg) else {
            return;
        };
        let new = match (offset, width) {
            (0, Width::W64) => self.arena.trunc(value, Width::W64),
            (0, Width::W32) => {
                let v = self.arena.trunc(value, Width::W32);
                self.arena.zext(v, Width::W64)
            }
            _ => {
                let old = self.state.reg(base);
                let shift = offset as u64 * 8;
                let keep_mask = !(width.mask() << shift);
                let km = self.arena.constant(keep_mask, Width::W64);
                let kept = self.arena.bin(BinOp::And, old, km);
                let v = self.arena.trunc(value, width);
                let v = self.arena.zext(v, Width::W64);
                let sh = self.arena.constant(shift, Width::W64);
                let placed = self.arena.bin(BinOp::Shl, v, sh);
                self.arena.bin(BinOp::Or, kept, placed)
            }
        };
        self.state.set_reg(base, new);
        // Note argument-register writes, so a guest call can tell which registers
        // were actually set up for it.
        if ARG_REGS.contains(&base) {
            self.written_args.insert(base);
        }
    }

    /// Effective address of a memory operand.
    ///
    /// The segment prefix is deliberately not folded in. FS/GS-relative accesses are
    /// boxed instead (see [`seg_relative`]), which re-emits the original instruction
    /// with its prefix intact, so adding the base here as well would apply it twice.
    fn mem_address(&mut self, inst: &Instruction) -> Ref {
        let mut acc = self
            .arena
            .constant(inst.memory_displacement64(), Width::W64);

        let base = inst.memory_base();
        if base == Register::RIP || base == Register::EIP {
            // iced already folded RIP into the displacement.
            return acc;
        }
        if base != Register::None {
            let b = self.read_reg(base);
            let b = self.arena.zext(b, Width::W64);
            acc = self.arena.bin(BinOp::Add, acc, b);
        }
        let index = inst.memory_index();
        if index != Register::None {
            let i = self.read_reg(index);
            let i = self.arena.zext(i, Width::W64);
            let scale = self
                .arena
                .constant(inst.memory_index_scale() as u64, Width::W64);
            let scaled = self.arena.bin(BinOp::Mul, i, scale);
            acc = self.arena.bin(BinOp::Add, acc, scaled);
        }
        acc
    }

    fn load(&mut self, addr: Ref, width: Width, site: u64) -> Ref {
        let addr = self.substitute(addr);
        if let Some(a) = self.arena.as_const(addr) {
            // Do not fold a read from a writable section. The loader or runtime may write a different value there before the code runs, so the file content is a placeholder rather than the real value.
            let writable = self.pe.is_writable(a);
            // BSS (uninitialized data) is an exception: it is in a writable section but is zero-initialized by the loader, so we can fold it to zero at load time.
            let is_bss = self.pe.is_bss(a);
            // An IAT slot is not writable by its section bits but *is* rewritten by the
            // loader, so it must be refused for the same reason a `.data` global is.
            let bound = self.pe.is_loader_bound(a, width.bytes() as u64);
            let pe = self.pe;
            let arena = &mut self.arena;
            let img = |x: u64| pe.image_u8(x);
            // Fold if: (not writable OR is BSS) AND not loader-bound
            // BSS is writable but we know it's zero at load time
            if (!writable || is_bss) && !bound {
                if let Some(v) = self.state.load_concrete(arena, a, width, img) {
                    return v;
                }
            }

            // Did not resolve.
            let v = self.arena.load(addr, width);
            if self.pe.image_u8(a).is_none() {
                self.unmapped_loads.push((site, a, width));
            }
            let region = self.region_of_addr(a);
            self.events.push(Event::Load {
                addr,
                value: v,
                width,
                site,
                region,
            });
            return v;
        }
        let v = self.arena.load(addr, width);
        self.symbolic_loads.push((site, width));
        // A symbolic address is only a guest access if it is rooted in guest state. A guest load computes its address from guest registers, so the expression reaches a parameter or an entry register.
        let region = self.region_of_symbolic(addr);
        self.events.push(Event::Load {
            addr,
            value: v,
            width,
            site,
            region,
        });
        v
    }

    /// Forget bytes at addr without recording a separate event.
    /// Boxed memory writes are re-emitted from their original encoding, so evaluation
    /// only needs to invalidate the affected state. The byte count supports operands
    /// wider than the IR's scalar widths.
    fn forget_memory(&mut self, addr: Ref, bytes: u32) {
        if let Some(a) = self.arena.as_const(addr) {
            self.state.forget_concrete(a, bytes);
            self.arena.bump_mem_gen();
        } else {
            // A symbolic address cannot be resolved against the byte map, so the
            // pending symbolic store is what a later load consults. Push an opaque
            // of the widest width that fits, which keeps the load from folding
            // against anything older at the same expression.
            let w = Width::from_bytes(bytes).unwrap_or(Width::W64);
            let v = self.arena.opaque("boxed_mem", w);
            self.state.sym_stores.push((addr, v, w));
            self.arena.bump_mem_gen();
        }
    }

    fn store(&mut self, addr: Ref, value: Ref, width: Width, site: u64) {
        if let Some(a) = self.arena.as_const(addr) {
            // Track the phantom-unwind pointer the VM parks in its frame for
            // SEH. It is a `.tvm0` pointer that never changes, so recording it
            // here keeps VIP detection from latching onto it.
            if width == Width::W64 {
                if let Some(v) = self.arena.as_const(value) {
                    let rbp = self.arena.as_const(self.state.reg(Reg::Rbp));
                    if rbp == Some(a) && self.in_vm(v) {
                        self.unwind_slot = Some(v);
                    }
                }
            }
            self.state.store_concrete(&self.arena, a, value, width);

            // Memory changed, so a later load of this address must not intern to
            // one taken before now. See `Op::Load`.
            self.arena.bump_mem_gen();

            let region = self.region_of_addr(a);
            self.events.push(Event::Store {
                addr,
                value,
                width,
                site,
                region,
            });
            return;
        }
        self.state.sym_stores.push((addr, value, width));
        self.arena.bump_mem_gen();
        let region = self.region_of_symbolic(addr);
        self.events.push(Event::Store {
            addr,
            value,
            width,
            site,
            region,
        });
    }

    // ! operands !

    fn op_width(&self, inst: &Instruction, op: u32) -> Width {
        let n = match inst.op_kind(op) {
            OpKind::Register => inst.op_register(op).size() as u32,
            OpKind::Memory => inst.memory_size().size() as u32,
            OpKind::Immediate8 | OpKind::Immediate8_2nd => 1,
            OpKind::Immediate16 | OpKind::Immediate8to16 => 2,
            OpKind::Immediate32 | OpKind::Immediate8to32 => 4,
            OpKind::Immediate64 | OpKind::Immediate8to64 | OpKind::Immediate32to64 => 8,
            _ => 8,
        };
        Width::from_bytes(n).unwrap_or(Width::W64)
    }

    fn read_op(&mut self, inst: &Instruction, op: u32) -> Ref {
        match inst.op_kind(op) {
            OpKind::Register => self.read_reg(inst.op_register(op)),
            OpKind::Memory => {
                let addr = self.mem_address(inst);
                let w = Width::from_bytes(inst.memory_size().size() as u32).unwrap_or(Width::W64);
                self.load(addr, w, inst.ip())
            }
            OpKind::Immediate8 => self.arena.constant(inst.immediate8() as u64, Width::W8),
            OpKind::Immediate8_2nd => self.arena.constant(inst.immediate8_2nd() as u64, Width::W8),
            OpKind::Immediate16 => self.arena.constant(inst.immediate16() as u64, Width::W16),
            OpKind::Immediate32 => self.arena.constant(inst.immediate32() as u64, Width::W32),
            OpKind::Immediate64 => self.arena.constant(inst.immediate64(), Width::W64),
            OpKind::Immediate8to16 => self
                .arena
                .constant(inst.immediate8to16() as u64, Width::W16),
            OpKind::Immediate8to32 => self
                .arena
                .constant(inst.immediate8to32() as u64, Width::W32),
            OpKind::Immediate8to64 => self
                .arena
                .constant(inst.immediate8to64() as u64, Width::W64),
            OpKind::Immediate32to64 => self
                .arena
                .constant(inst.immediate32to64() as u64, Width::W64),
            _ => self.arena.opaque("operand", Width::W64),
        }
    }

    fn write_op(&mut self, inst: &Instruction, op: u32, value: Ref) {
        match inst.op_kind(op) {
            OpKind::Register => self.write_reg(inst.op_register(op), value),
            OpKind::Memory => {
                let addr = self.mem_address(inst);
                let w = Width::from_bytes(inst.memory_size().size() as u32).unwrap_or(Width::W64);
                self.store(addr, value, w, inst.ip());
            }
            _ => {}
        }
    }

    // ! stack !

    fn push(&mut self, value: Ref, width: Width, site: u64) {
        let sp = self.state.reg(Reg::Rsp);
        let delta = self
            .arena
            .constant((width.bytes() as u64).wrapping_neg(), Width::W64);
        let new_sp = self.arena.bin(BinOp::Add, sp, delta);
        self.state.set_reg(Reg::Rsp, new_sp);
        self.store(new_sp, value, width, site);
    }

    fn pop(&mut self, width: Width, site: u64) -> Ref {
        let sp = self.state.reg(Reg::Rsp);
        let v = self.load(sp, width, site);
        let delta = self.arena.constant(width.bytes() as u64, Width::W64);
        let new_sp = self.arena.bin(BinOp::Add, sp, delta);
        self.state.set_reg(Reg::Rsp, new_sp);
        v
    }

    // ! flags !

    fn set_zf_sf_pf(&mut self, result: Ref, width: Width) {
        let zero = self.arena.constant(0, width);
        let zf = self.arena.bin(BinOp::Eq, result, zero);
        self.state.set_flag(Flag::Zf, zf);

        let sb = self.arena.constant(width.sign_bit(), width);
        let masked = self.arena.bin(BinOp::And, result, sb);
        let sf = self.arena.bin(BinOp::Eq, masked, sb);
        self.state.set_flag(Flag::Sf, sf);

        let low = self.arena.trunc(result, Width::W8);
        let pf = self.arena.un(UnOp::ParityByte, low);
        self.state.set_flag(Flag::Pf, pf);
    }

    fn set_logic_flags(&mut self, result: Ref, width: Width) {
        self.set_zf_sf_pf(result, width);
        let zero = self.arena.constant(0, Width::W8);
        self.state.set_flag(Flag::Cf, zero);
        self.state.set_flag(Flag::Of, zero);
        let undef = self.arena.undef("af_undef", Width::W8);
        self.state.set_flag(Flag::Af, undef);
    }

    /// Flags for `a + b` (and `adc` when `carry_in` is supplied).
    fn set_add_flags(&mut self, a: Ref, b: Ref, result: Ref, width: Width) {
        self.set_zf_sf_pf(result, width);
        // CF: unsigned overflow, result < a
        let cf = self.arena.bin(BinOp::Ult, result, a);
        self.state.set_flag(Flag::Cf, cf);
        // OF: (a^result) & (b^result) & sign
        let x1 = self.arena.bin(BinOp::Xor, a, result);
        let x2 = self.arena.bin(BinOp::Xor, b, result);
        let both = self.arena.bin(BinOp::And, x1, x2);
        let sb = self.arena.constant(width.sign_bit(), width);
        let m = self.arena.bin(BinOp::And, both, sb);
        let of = self.arena.bin(BinOp::Eq, m, sb);
        self.state.set_flag(Flag::Of, of);
        self.set_af(a, b, result, width);
    }

    fn set_sub_flags(&mut self, a: Ref, b: Ref, result: Ref, width: Width) {
        self.set_zf_sf_pf(result, width);
        let cf = self.arena.bin(BinOp::Ult, a, b);
        self.state.set_flag(Flag::Cf, cf);
        // OF: (a^b) & (a^result) & sign
        let x1 = self.arena.bin(BinOp::Xor, a, b);
        let x2 = self.arena.bin(BinOp::Xor, a, result);
        let both = self.arena.bin(BinOp::And, x1, x2);
        let sb = self.arena.constant(width.sign_bit(), width);
        let m = self.arena.bin(BinOp::And, both, sb);
        let of = self.arena.bin(BinOp::Eq, m, sb);
        self.state.set_flag(Flag::Of, of);
        self.set_af(a, b, result, width);
    }

    fn set_af(&mut self, a: Ref, b: Ref, result: Ref, width: Width) {
        let x = self.arena.bin(BinOp::Xor, a, b);
        let x = self.arena.bin(BinOp::Xor, x, result);
        let m = self.arena.constant(0x10, width);
        let m2 = self.arena.bin(BinOp::And, x, m);
        let af = self.arena.bin(BinOp::Eq, m2, m);
        self.state.set_flag(Flag::Af, af);
    }

    /// Evaluate a condition code into a 0/1 value.
    fn condition(&mut self, cc: ConditionCode) -> Ref {
        let one = self.arena.constant(1, Width::W8);
        let f = |s: &mut Self, fl: Flag| s.state.flag(fl);
        match cc {
            ConditionCode::O => f(self, Flag::Of),
            ConditionCode::No => {
                let v = f(self, Flag::Of);
                self.arena.bin(BinOp::Xor, v, one)
            }
            ConditionCode::B => f(self, Flag::Cf),
            ConditionCode::Ae => {
                let v = f(self, Flag::Cf);
                self.arena.bin(BinOp::Xor, v, one)
            }
            ConditionCode::E => f(self, Flag::Zf),
            ConditionCode::Ne => {
                let v = f(self, Flag::Zf);
                self.arena.bin(BinOp::Xor, v, one)
            }
            ConditionCode::Be => {
                let c = f(self, Flag::Cf);
                let z = f(self, Flag::Zf);
                self.arena.bin(BinOp::Or, c, z)
            }
            ConditionCode::A => {
                let c = f(self, Flag::Cf);
                let z = f(self, Flag::Zf);
                let o = self.arena.bin(BinOp::Or, c, z);
                self.arena.bin(BinOp::Xor, o, one)
            }
            ConditionCode::S => f(self, Flag::Sf),
            ConditionCode::Ns => {
                let v = f(self, Flag::Sf);
                self.arena.bin(BinOp::Xor, v, one)
            }
            ConditionCode::P => f(self, Flag::Pf),
            ConditionCode::Np => {
                let v = f(self, Flag::Pf);
                self.arena.bin(BinOp::Xor, v, one)
            }
            ConditionCode::L => {
                let s = f(self, Flag::Sf);
                let o = f(self, Flag::Of);
                self.arena.bin(BinOp::Xor, s, o)
            }
            ConditionCode::Ge => {
                let s = f(self, Flag::Sf);
                let o = f(self, Flag::Of);
                let x = self.arena.bin(BinOp::Xor, s, o);
                self.arena.bin(BinOp::Xor, x, one)
            }
            ConditionCode::Le => {
                let s = f(self, Flag::Sf);
                let o = f(self, Flag::Of);
                let z = f(self, Flag::Zf);
                let x = self.arena.bin(BinOp::Xor, s, o);
                self.arena.bin(BinOp::Or, x, z)
            }
            ConditionCode::G => {
                let s = f(self, Flag::Sf);
                let o = f(self, Flag::Of);
                let z = f(self, Flag::Zf);
                let x = self.arena.bin(BinOp::Xor, s, o);
                let or = self.arena.bin(BinOp::Or, x, z);
                self.arena.bin(BinOp::Xor, or, one)
            }
        }
    }

    // ! stepping !

    fn fallthrough_is_trap(&self, addr: u64) -> bool {
        disasm::decode_at(self.pe, addr).is_some_and(|i| i.mnemonic() == Mnemonic::Int3)
    }

    pub fn trace_tail(&self, n: usize) -> Vec<u64> {
        let skip = self.trace.len().saturating_sub(n);
        self.trace.iter().skip(skip).copied().collect()
    }

    pub fn step(&mut self, ip: u64) -> Step {
        let Some(inst) = disasm::decode_at(self.pe, ip) else {
            return Step::Stopped(Stop::Unreadable { site: ip });
        };
        self.steps += 1;
        if self.trace.len() == TRACE_TAIL_CAP {
            self.trace.pop_front();
        }
        self.trace.push_back(ip);
        let next = inst.next_ip();

        use Mnemonic as M;

        // FS/GS-relative accesses are boxed, whatever the mnemonic.
        if self.box_unmodelled && seg_relative(&inst) && is_boxable(inst.mnemonic()) {
            return self.box_instruction(&inst, ip, next);
        }

        match inst.mnemonic() {
            M::Nop | M::Pause | M::Prefetchw | M::Prefetchnta => Step::Next(next),

            M::Mov => {
                let v = self.read_op(&inst, 1);
                self.write_op(&inst, 0, v);
                Step::Next(next)
            }
            M::Movzx => {
                let v = self.read_op(&inst, 1);
                let dst = self.op_width(&inst, 0);
                let z = self.arena.zext(v, dst);
                self.write_op(&inst, 0, z);
                Step::Next(next)
            }
            M::Movsx | M::Movsxd => {
                let v = self.read_op(&inst, 1);
                let dst = self.op_width(&inst, 0);
                let z = self.arena.sext(v, dst);
                self.write_op(&inst, 0, z);
                Step::Next(next)
            }
            M::Lea => {
                let addr = self.mem_address(&inst);
                let w = self.op_width(&inst, 0);
                let v = self.arena.trunc(addr, w);
                self.write_op(&inst, 0, v);
                Step::Next(next)
            }
            M::Xchg => {
                let a = self.read_op(&inst, 0);
                let b = self.read_op(&inst, 1);
                self.write_op(&inst, 0, b);
                self.write_op(&inst, 1, a);
                Step::Next(next)
            }
            M::Cmpxchg => {
                // Only the non-atomic data flow matters here.
                let dst = self.read_op(&inst, 0);
                let src = self.read_op(&inst, 1);
                let w = self.op_width(&inst, 0);
                let acc_reg = acc_for_width(w);
                let acc = self.read_reg(acc_reg);
                let eq = self.arena.bin(BinOp::Eq, acc, dst);
                let newdst = self.arena.select(eq, src, dst);
                let newacc = self.arena.select(eq, acc, dst);
                let sub = self.arena.bin(BinOp::Sub, acc, dst);
                self.set_sub_flags(acc, dst, sub, w);
                self.write_op(&inst, 0, newdst);
                self.write_reg(acc_reg, newacc);
                Step::Next(next)
            }

            M::Push => {
                let v = self.read_op(&inst, 0);
                let w = match inst.op_kind(0) {
                    OpKind::Register => self.op_width(&inst, 0),
                    OpKind::Memory => {
                        Width::from_bytes(inst.memory_size().size() as u32).unwrap_or(Width::W64)
                    }
                    _ => Width::W64,
                };
                // Immediates and 32-bit forms still move RSP by 8 in long mode.
                let v = self.arena.sext(v, Width::W64);
                let _ = w;
                self.push(v, Width::W64, ip);
                Step::Next(next)
            }
            M::Pop => {
                let v = self.pop(Width::W64, ip);
                self.write_op(&inst, 0, v);
                Step::Next(next)
            }
            M::Pushfq => {
                let v = self.state.pack_flags(&mut self.arena);
                self.push(v, Width::W64, ip);
                Step::Next(next)
            }
            M::Popfq => {
                let v = self.pop(Width::W64, ip);
                self.state.unpack_flags(&mut self.arena, v);
                Step::Next(next)
            }

            M::Add | M::Adc => {
                let w = self.op_width(&inst, 0);
                let a = self.read_op(&inst, 0);
                let mut b = self.read_op(&inst, 1);
                b = self.arena.sext(b, w);
                if inst.mnemonic() == M::Adc {
                    let cf = self.state.flag(Flag::Cf);
                    let cf = self.arena.zext(cf, w);
                    b = self.arena.bin(BinOp::Add, b, cf);
                }
                let r = self.arena.bin(BinOp::Add, a, b);
                let r = self.arena.trunc(r, w);
                self.set_add_flags(a, b, r, w);
                self.write_op(&inst, 0, r);
                Step::Next(next)
            }
            M::Sub | M::Sbb => {
                let w = self.op_width(&inst, 0);
                let a = self.read_op(&inst, 0);
                let mut b = self.read_op(&inst, 1);
                b = self.arena.sext(b, w);
                if inst.mnemonic() == M::Sbb {
                    let cf = self.state.flag(Flag::Cf);
                    let cf = self.arena.zext(cf, w);
                    b = self.arena.bin(BinOp::Add, b, cf);
                }
                let r = self.arena.bin(BinOp::Sub, a, b);
                let r = self.arena.trunc(r, w);
                self.set_sub_flags(a, b, r, w);
                self.write_op(&inst, 0, r);
                Step::Next(next)
            }
            M::Cmp => {
                let w = self.op_width(&inst, 0);
                let a = self.read_op(&inst, 0);
                let b = self.read_op(&inst, 1);
                let b = self.arena.sext(b, w);
                let r = self.arena.bin(BinOp::Sub, a, b);
                let r = self.arena.trunc(r, w);
                self.set_sub_flags(a, b, r, w);
                Step::Next(next)
            }
            M::Test => {
                let w = self.op_width(&inst, 0);
                let a = self.read_op(&inst, 0);
                let b = self.read_op(&inst, 1);
                let b = self.arena.sext(b, w);
                let r = self.arena.bin(BinOp::And, a, b);
                let r = self.arena.trunc(r, w);
                self.set_logic_flags(r, w);
                Step::Next(next)
            }
            M::And | M::Or | M::Xor => {
                let w = self.op_width(&inst, 0);
                let a = self.read_op(&inst, 0);
                let b = self.read_op(&inst, 1);
                let b = self.arena.sext(b, w);
                let op = match inst.mnemonic() {
                    M::And => BinOp::And,
                    M::Or => BinOp::Or,
                    _ => BinOp::Xor,
                };
                let r = self.arena.bin(op, a, b);
                let r = self.arena.trunc(r, w);
                self.set_logic_flags(r, w);
                self.write_op(&inst, 0, r);
                Step::Next(next)
            }
            M::Not => {
                let a = self.read_op(&inst, 0);
                let r = self.arena.not(a);
                self.write_op(&inst, 0, r);
                Step::Next(next)
            }
            M::Neg => {
                let w = self.op_width(&inst, 0);
                let a = self.read_op(&inst, 0);
                let zero = self.arena.constant(0, w);
                let r = self.arena.bin(BinOp::Sub, zero, a);
                let r = self.arena.trunc(r, w);
                self.set_sub_flags(zero, a, r, w);
                self.write_op(&inst, 0, r);
                Step::Next(next)
            }
            M::Inc | M::Dec => {
                let w = self.op_width(&inst, 0);
                let a = self.read_op(&inst, 0);
                let one = self.arena.constant(1, w);
                let (r, is_inc) = if inst.mnemonic() == M::Inc {
                    (self.arena.bin(BinOp::Add, a, one), true)
                } else {
                    (self.arena.bin(BinOp::Sub, a, one), false)
                };
                let r = self.arena.trunc(r, w);
                // INC/DEC preserve CF.
                let cf = self.state.flag(Flag::Cf);
                if is_inc {
                    self.set_add_flags(a, one, r, w);
                } else {
                    self.set_sub_flags(a, one, r, w);
                }
                self.state.set_flag(Flag::Cf, cf);
                self.write_op(&inst, 0, r);
                Step::Next(next)
            }

            M::Shl | M::Shr | M::Sar | M::Rol | M::Ror => {
                let w = self.op_width(&inst, 0);
                let a = self.read_op(&inst, 0);
                let cnt = if inst.op_count() > 1 {
                    self.read_op(&inst, 1)
                } else {
                    self.arena.constant(1, Width::W8)
                };
                let cnt = self.arena.zext(cnt, w);
                let op = match inst.mnemonic() {
                    M::Shl => BinOp::Shl,
                    M::Shr => BinOp::Shr,
                    M::Sar => BinOp::Sar,
                    M::Rol => BinOp::Rol,
                    _ => BinOp::Ror,
                };
                let r = self.arena.bin(op, a, cnt);
                let r = self.arena.trunc(r, w);
                // Shift flags are only approximated: ZF/SF/PF are exact, and
                // CF/OF are marked unknown unless the count folded to 0.
                match self.arena.as_const(cnt) {
                    Some(0) => {}
                    _ => {
                        self.set_zf_sf_pf(r, w);
                        let u = self.arena.opaque("shift_cf", Width::W8);
                        self.state.set_flag(Flag::Cf, u);
                        let u = self.arena.opaque("shift_of", Width::W8);
                        self.state.set_flag(Flag::Of, u);
                    }
                }
                self.write_op(&inst, 0, r);
                Step::Next(next)
            }
            M::Bswap => {
                let a = self.read_op(&inst, 0);
                let r = self.arena.un(UnOp::Bswap, a);
                self.write_op(&inst, 0, r);
                Step::Next(next)
            }
            M::Bt => {
                let w = self.op_width(&inst, 0);
                let a = self.read_op(&inst, 0);
                let b = self.read_op(&inst, 1);
                let b = self.arena.zext(b, w);
                let sh = self.arena.bin(BinOp::Shr, a, b);
                let one = self.arena.constant(1, w);
                let bit = self.arena.bin(BinOp::And, sh, one);
                let cf = self.arena.trunc(bit, Width::W8);
                self.state.set_flag(Flag::Cf, cf);
                Step::Next(next)
            }
            M::Cdq | M::Cqo | M::Cwd => {
                let w = match inst.mnemonic() {
                    M::Cqo => Width::W64,
                    M::Cdq => Width::W32,
                    _ => Width::W16,
                };
                let acc = self.read_reg(acc_for_width(w));
                let sb = self.arena.constant(w.sign_bit(), w);
                let m = self.arena.bin(BinOp::And, acc, sb);
                let is_neg = self.arena.bin(BinOp::Eq, m, sb);
                let all = self.arena.constant(w.mask(), w);
                let zero = self.arena.constant(0, w);
                let ext = self.arena.select(is_neg, all, zero);
                self.write_reg(dx_for_width(w), ext);
                Step::Next(next)
            }
            M::Imul => {
                let w = self.op_width(&inst, 0);
                match inst.op_count() {
                    1 => {
                        let a = self.read_reg(acc_for_width(w));
                        let b = self.read_op(&inst, 0);
                        let lo = self.arena.bin(BinOp::Mul, a, b);
                        let hi = self.arena.bin(BinOp::MulHiS, a, b);
                        self.write_reg(acc_for_width(w), lo);
                        if w != Width::W8 {
                            self.write_reg(dx_for_width(w), hi);
                        }
                        self.clobber_arith_flags();
                    }
                    2 => {
                        let a = self.read_op(&inst, 0);
                        let b = self.read_op(&inst, 1);
                        let b = self.arena.sext(b, w);
                        let r = self.arena.bin(BinOp::Mul, a, b);
                        let r = self.arena.trunc(r, w);
                        self.write_op(&inst, 0, r);
                        self.set_zf_sf_pf(r, w);
                        self.clobber_cf_of();
                    }
                    _ => {
                        let a = self.read_op(&inst, 1);
                        let b = self.read_op(&inst, 2);
                        let b = self.arena.sext(b, w);
                        let r = self.arena.bin(BinOp::Mul, a, b);
                        let r = self.arena.trunc(r, w);
                        self.write_op(&inst, 0, r);
                        self.set_zf_sf_pf(r, w);
                        self.clobber_cf_of();
                    }
                }
                Step::Next(next)
            }
            M::Mul => {
                let w = self.op_width(&inst, 0);
                let a = self.read_reg(acc_for_width(w));
                let b = self.read_op(&inst, 0);
                let lo = self.arena.bin(BinOp::Mul, a, b);
                let hi = self.arena.bin(BinOp::MulHiU, a, b);
                self.write_reg(acc_for_width(w), lo);
                if w != Width::W8 {
                    self.write_reg(dx_for_width(w), hi);
                }
                self.clobber_arith_flags();
                Step::Next(next)
            }
            M::Div | M::Idiv => {
                let w = self.op_width(&inst, 0);
                let lo = self.read_reg(acc_for_width(w));
                let b = self.read_op(&inst, 0);

                // x86 DIV/IDIV uses a double-width dividend:
                // - 8-bit: AX (16-bit) ÷ operand → AL (quotient), AH (remainder)
                // - 16-bit: DX:AX (32-bit) ÷ operand → AX, DX
                // - 32-bit: EDX:EAX (64-bit) ÷ operand → EAX, EDX
                // - 64-bit: RDX:RAX (128-bit) ÷ operand → RAX, RDX
                let dividend = if w == Width::W8 {
                    // 8-bit form: dividend is 16-bit AX
                    lo
                } else {
                    // Combine high:low into double-width value
                    let hi = self.read_reg(dx_for_width(w));
                    // Widen both halves to double-width for the shift
                    let double_w = match w {
                        Width::W16 => Width::W32,
                        Width::W32 => Width::W64,
                        Width::W64 => Width::W64, // Can't represent 128-bit, approximate
                        Width::W8 => Width::W16,
                    };
                    let hi_wide = self.arena.zext(hi, double_w);
                    let lo_wide = self.arena.zext(lo, double_w);
                    let shift_amt = self.arena.constant(w.bits() as u64, double_w);
                    let hi_shifted = self.arena.bin(BinOp::Shl, hi_wide, shift_amt);
                    self.arena.bin(BinOp::Or, hi_shifted, lo_wide)
                };

                let (dop, rop) = if inst.mnemonic() == M::Div {
                    (BinOp::UDiv, BinOp::URem)
                } else {
                    (BinOp::SDiv, BinOp::SRem)
                };
                let q = self.arena.bin(dop, dividend, b);
                let r = self.arena.bin(rop, dividend, b);

                if w == Width::W8 {
                    // 8-bit: quotient in AL, remainder in AH (both in AX)
                    let q_byte = self.arena.trunc(q, Width::W8);
                    let r_byte = self.arena.trunc(r, Width::W8);
                    let q_wide = self.arena.zext(q_byte, Width::W16);
                    let r_wide = self.arena.zext(r_byte, Width::W16);
                    let eight = self.arena.constant(8, Width::W16);
                    let r_shifted = self.arena.bin(BinOp::Shl, r_wide, eight);
                    let ax = self.arena.bin(BinOp::Or, q_wide, r_shifted);
                    self.write_reg(Register::AX, ax);
                } else {
                    // Truncate results back to original width
                    let q_trunc = self.arena.trunc(q, w);
                    let r_trunc = self.arena.trunc(r, w);
                    self.write_reg(acc_for_width(w), q_trunc);
                    self.write_reg(dx_for_width(w), r_trunc);
                }
                self.clobber_arith_flags();
                Step::Next(next)
            }

            M::Cmovo
            | M::Cmovno
            | M::Cmovb
            | M::Cmovae
            | M::Cmove
            | M::Cmovne
            | M::Cmovbe
            | M::Cmova
            | M::Cmovs
            | M::Cmovns
            | M::Cmovp
            | M::Cmovnp
            | M::Cmovl
            | M::Cmovge
            | M::Cmovle
            | M::Cmovg => {
                let cc = cmov_cc(inst.mnemonic()).expect("cmov cc");
                let cond = self.condition(cc);
                let a = self.read_op(&inst, 0);
                let b = self.read_op(&inst, 1);
                let r = self.arena.select(cond, b, a);
                self.write_op(&inst, 0, r);
                Step::Next(next)
            }
            M::Seto
            | M::Setno
            | M::Setb
            | M::Setae
            | M::Sete
            | M::Setne
            | M::Setbe
            | M::Seta
            | M::Sets
            | M::Setns
            | M::Setp
            | M::Setnp
            | M::Setl
            | M::Setge
            | M::Setle
            | M::Setg => {
                let cc = setcc_cc(inst.mnemonic()).expect("setcc cc");
                let cond = self.condition(cc);
                let v = self.arena.trunc(cond, Width::W8);
                self.write_op(&inst, 0, v);
                Step::Next(next)
            }

            M::Jmp => {
                if let Some(t) = disasm::direct_target(&inst) {
                    return Step::Next(t);
                }
                // A tail call through the IAT: `jmp qword ptr [slot]`
                // The callee returns straight to caller, so this is the last instruction of the guest func
                if inst.op_kind(0) == OpKind::Memory {
                    let slot_addr = self.mem_address(&inst);
                    if let Some(slot) = self.arena.as_const(slot_addr) {
                        if is_iat_tail_call(slot, &self.import_slots) {
                            let rsp = self.state.concrete_rsp(&self.arena);
                            let args = self.call_arguments();
                            // Before the push, so the event can name the opaque.
                            let ret = self.clobber_call();
                            self.events.push(Event::Call {
                                target: None,
                                site: ip,
                                rsp,
                                import_slot: Some(slot),
                                args,
                                ret,
                            });
                            if self.noreturn_slots.contains(&slot) {
                                return Step::Stopped(Stop::NoReturn { site: ip });
                            }
                            let dest = self.pop(Width::W64, ip);
                            return Step::Stopped(Stop::Return { site: ip, dest });
                        }
                    }
                }
                let dest = self.read_op(&inst, 0);
                let dest = self.substitute(dest);
                match self.arena.as_const(dest) {
                    Some(t) => {
                        // A resolved indirect jump is a VM handler transition.
                        self.dispatches.push((ip, t));
                        // Reaching the identity of an already-recovered block
                        // closes a loop. Stopping keeps cost linear in the
                        // number of distinct blocks; continuing would unroll the
                        // loop once per iteration.
                        if self.stop_on_revisit {
                            let vip = self.vip_at_entry(t);
                            let progressed = self.guest_event_count() > self.events_at_block_entry;
                            if progressed && self.block_entries.contains(&(t, vip)) {
                                return Step::Stopped(Stop::Backedge {
                                    site: ip,
                                    target: t,
                                    vip,
                                });
                            }
                        }
                        Step::Next(t)
                    }
                    None => Step::Stopped(Stop::SymbolicBranch { site: ip, dest }),
                }
            }
            M::Jo
            | M::Jno
            | M::Jb
            | M::Jae
            | M::Je
            | M::Jne
            | M::Jbe
            | M::Ja
            | M::Js
            | M::Jns
            | M::Jp
            | M::Jnp
            | M::Jl
            | M::Jge
            | M::Jle
            | M::Jg => {
                let cc = jcc_cc(inst.mnemonic()).expect("jcc cc");
                let cond = self.condition(cc);
                let cond = self.substitute(cond);
                let target = disasm::direct_target(&inst).unwrap_or(next);
                match self.arena.as_const(cond) {
                    Some(0) => Step::Next(next),
                    Some(_) => Step::Next(target),
                    // A *native* conditional branch on a guest value. Both edges are already known statically, so this is reported separately from a computed jump: there is nothing to fold, CFG just needs to fork on the predicate.
                    None => Step::Stopped(Stop::NativeBranch {
                        site: ip,
                        predicate: cond,
                        taken: target,
                        not_taken: next,
                    }),
                }
            }
            M::Call => {
                // The VM enters and threads through its dispatcher with CALLs
                // that never return. Two signals identify them:
                //   1. the call target is inside the VM section, and
                //   2. an `int3` (or a decoy `jmp` to the same target) follows.
                // Signal 1 is the reliable one: guest "boxed" instructions and
                // semantic calls always leave the VM section, VM control flow
                // never does.
                let ret = next;
                // An indirect call through a fixed address is a call through the import address table.
                let mut import_slot = None;
                if disasm::direct_target(&inst).is_none() && inst.op_kind(0) == OpKind::Memory {
                    let a = self.mem_address(&inst);
                    if let Some(c) = self.arena.as_const(a) {
                        if !self.pe.is_executable(c) {
                            import_slot = Some(c);
                        }
                    }
                }
                let target = disasm::direct_target(&inst).or_else(|| {
                    let d = self.read_op(&inst, 0);
                    self.arena.as_const(d)
                });
                // A target that is not executable code was never a callee: it is an
                // unbound import entry or a misfolded value. Discarding it keeps the
                // event honest instead of naming a bogus address.
                let target = target.filter(|t| self.pe.is_executable(*t));
                let internal = target.is_some_and(|t| self.in_vm(t));
                if internal {
                    let r = self.arena.constant(ret, Width::W64);
                    self.push(r, Width::W64, ip);
                    return Step::Next(target.expect("internal implies known target"));
                }
                // Guest call: record it, skip over it, and mark the volatile
                // register set as clobbered per the Windows x64 ABI.
                let t = target.map(|t| self.arena.constant(t, Width::W64));
                let rsp = self.state.concrete_rsp(&self.arena);
                // Keep arguments explicitly established by this run before applying
                // ABI clobbers.
                let args = self.call_arguments();
                let ret = self.clobber_call();
                self.events.push(Event::Call {
                    target: t,
                    site: ip,
                    rsp,
                    import_slot,
                    args,
                    ret,
                });
                // A noreturn call ends the block; trailing padding or traps are not
                // part of the guest path.
                if import_slot.is_some_and(|s| self.noreturn_slots.contains(&s))
                    || self.fallthrough_is_trap(next)
                    || target.is_some_and(|t| callee_never_returns(self.pe, t))
                {
                    return Step::Stopped(Stop::NoReturn { site: ip });
                }
                Step::Next(next)
            }
            M::Ret => {
                let dest = self.pop(Width::W64, ip);
                if inst.op_count() > 0 {
                    if let Some(imm) = Width::from_bytes(1) {
                        let _ = imm;
                    }
                    let n = inst.immediate16() as u64;
                    let sp = self.state.reg(Reg::Rsp);
                    let k = self.arena.constant(n, Width::W64);
                    let sp = self.arena.bin(BinOp::Add, sp, k);
                    self.state.set_reg(Reg::Rsp, sp);
                }
                Step::Stopped(Stop::Return { site: ip, dest })
            }

            // CET feature probe. `rdsspq` retires as a NOP without shadow stacks, leaving the destination untouched; TVM zeroes the register first and tests it afterwards.
            M::Rdsspq | M::Rdsspd => {
                if !self.assume_cet_disabled {
                    let v = self.arena.opaque("rdssp", Width::W64);
                    self.write_op(&inst, 0, v);
                }
                Step::Next(next)
            }
            M::Incsspq | M::Incsspd => Step::Next(next),

            // Boxed / unmodelled but semantically inert for our purposes.
            M::Int3 => Step::Stopped(Stop::Unsupported {
                site: ip,
                text: "int3".into(),
            }),

            other => {
                // SSE data movement, modelled rather than boxed.
                if self.try_sse_move(&inst, other, ip) {
                    return Step::Next(next);
                }

                let mut fmt = iced_x86::IntelFormatter::new();
                let mut s = String::new();
                iced_x86::Formatter::format(&mut fmt, &inst, &mut s);

                // Boxed instructions: anything outside the subset of AMD64 the VM models is executed natively, with the VM restoring guest state around it.
                if self.box_unmodelled && is_boxable(other) {
                    return self.box_instruction(&inst, ip, next);
                }

                Step::Stopped(Stop::Unsupported { site: ip, text: s })
            }
        }
    }

    /// Record an instruction verbatim and make its effects opaque. The instruction is not modelled: it is kept as bytes so lowering can re-emit it, and its register/flag results become opaques so nothing downstream assumes a stale value.
    fn box_instruction(&mut self, inst: &Instruction, ip: u64, next: u64) -> Step {
        let mut fmt = iced_x86::IntelFormatter::new();
        let mut text = String::new();
        iced_x86::Formatter::format(&mut fmt, inst, &mut text);

        // Record the memory operand symbolically. The encoding names the VM's registers, which mean nothing in the emitted function, so the address has to travel with the instruction.
        let mem = if !seg_relative(inst)
            && inst.op_count() > 0
            && (0..inst.op_count()).any(|i| inst.op_kind(i) == OpKind::Memory)
        {
            let writes = inst.op_kind(0) == OpKind::Memory;
            Some(BoxedMem {
                addr: self.mem_address(inst),
                bytes: inst.memory_size().size() as u32,
                writes,
            })
        } else {
            None
        };
        let bytes = self
            .pe
            .read_va(ip, inst.len())
            .map(|b| b.to_vec())
            .unwrap_or_default();
        let writes_mem = inst.op_count() > 0 && inst.op_kind(0) == OpKind::Memory;
        // Inputs *before* effects: `apply_boxed_effects` replaces the written registers
        // Read implicit inputs before replacing architectural outputs with opaques.
        // For instructions such as `cpuid`, reading later would capture the new output.
        // writes it; reading afterwards would capture the opaque it just created
        // instead of the guest value the instruction actually consumes.
        let uses = self.boxed_uses(inst);
        // Effects first: the event carries the opaques this produces, so scheduling can
        // place them at the instruction rather than at their first reader.
        let defs = self.apply_boxed_effects(inst, inst.mnemonic(), writes_mem);
        self.events.push(Event::Boxed {
            site: ip,
            text,
            bytes,
            mem,
            defs,
            uses,
        });
        Step::Next(next)
    }

    /// Rebuild `r` with every pinned leaf replaced by its concrete value.
    ///
    /// Substitution runs at use time because pins are discovered after expressions
    /// become symbolic. A watermark skips nodes that predate the pins, and the memo is
    /// retained for the current pin set.
    pub fn substitute(&mut self, r: Ref) -> Ref {
        if self.pins.is_empty() || self.arena.is_const(r) {
            return r;
        }
        if self.pins.len() != self.memo_pin_count {
            self.subst_memo.clear();
            self.pin_dep.clear();
            self.memo_pin_count = self.pins.len();
        }
        let mut memo = std::mem::take(&mut self.subst_memo);
        let out = self.subst_rec(r, &mut memo);
        self.subst_memo = memo;
        out
    }

    fn subst_rec(&mut self, r: Ref, memo: &mut std::collections::HashMap<Ref, Ref>) -> Ref {
        if let Some(&v) = memo.get(&r) {
            return v;
        }
        // Constants can never contain a pinned leaf, and neither can anything
        // that does not reach one. `depends_on_pins` answers that in one cached
        // pass, which turns substitution from a full DAG rewrite into a walk of
        // just the affected cone.
        if !self.depends_on_pins(r) {
            memo.insert(r, r);
            return r;
        }
        if let Some(&(_, v)) = self.pins.iter().find(|(p, _)| *p == r) {
            let w = self.arena.width(r);
            let k = self.arena.constant(v, w);
            memo.insert(r, k);
            return k;
        }
        let out = match *self.arena.op(r) {
            crate::ir::expr::Op::Const(_)
            | crate::ir::expr::Op::InitReg(_)
            | crate::ir::expr::Op::Opaque(..)
            | crate::ir::expr::Op::Param(..) => r,
            crate::ir::expr::Op::Load(addr, gen_at) => {
                let a = self.subst_rec(addr, memo);
                if a == addr {
                    r
                } else {
                    let w = self.arena.width(r);
                    // Re-attempt resolution against image/abstract memory now that the address may have become concrete. This must not go through `load`, which records events: substitution is a pure rewrite and may be run many times over the same node.
                    if let Some(c) = self.arena.as_const(a) {
                        let is_bss = self.pe.is_bss(c);
                        if (!self.pe.is_writable(c) || is_bss)
                            && !self.pe.is_loader_bound(c, w.bytes() as u64)
                        {
                            let pe = self.pe;
                            let img = |x: u64| pe.image_u8(x);
                            if let Some(v) = self.state.load_concrete(&mut self.arena, c, w, img) {
                                return v;
                            }
                        }
                    }
                    // Preserve the generation: substitution is a rewrite of an
                    // existing load, not a new one, and stamping the current
                    // generation would merge it with unrelated loads.
                    self.arena.load_at(a, w, gen_at)
                }
            }
            crate::ir::expr::Op::Bin(op, x, y) => {
                let (nx, ny) = (self.subst_rec(x, memo), self.subst_rec(y, memo));
                if nx == x && ny == y {
                    r
                } else {
                    self.arena.bin(op, nx, ny)
                }
            }
            crate::ir::expr::Op::Un(op, x) => {
                let nx = self.subst_rec(x, memo);
                if nx == x { r } else { self.arena.un(op, nx) }
            }
            crate::ir::expr::Op::Zext(x) => {
                let nx = self.subst_rec(x, memo);
                let w = self.arena.width(r);
                if nx == x { r } else { self.arena.zext(nx, w) }
            }
            crate::ir::expr::Op::Sext(x) => {
                let nx = self.subst_rec(x, memo);
                let w = self.arena.width(r);
                if nx == x { r } else { self.arena.sext(nx, w) }
            }
            crate::ir::expr::Op::Trunc(x) => {
                let nx = self.subst_rec(x, memo);
                let w = self.arena.width(r);
                if nx == x { r } else { self.arena.trunc(nx, w) }
            }
            crate::ir::expr::Op::Select(c, x, y) => {
                let nc = self.subst_rec(c, memo);
                let nx = self.subst_rec(x, memo);
                let ny = self.subst_rec(y, memo);
                if nc == c && nx == x && ny == y {
                    r
                } else {
                    self.arena.select(nc, nx, ny)
                }
            }
        };
        memo.insert(r, out);
        out
    }

    /// Whether `r` transitively reaches any pinned leaf.
    ///
    /// Cached per (node, pin-set). Most of the DAG is guest semantics that has
    /// nothing to do with the pinned flag bit, so answering this cheaply is what
    /// makes pin-and-continue affordable.
    fn depends_on_pins(&mut self, r: Ref) -> bool {
        if let Some(&b) = self.pin_dep.get(&r) {
            return b;
        }
        let out = if self.pins.iter().any(|(p, _)| *p == r) {
            true
        } else {
            match *self.arena.op(r) {
                crate::ir::expr::Op::Const(_)
                | crate::ir::expr::Op::InitReg(_)
                | crate::ir::expr::Op::Opaque(..)
                | crate::ir::expr::Op::Param(..) => false,
                crate::ir::expr::Op::Load(a, _) => self.depends_on_pins(a),
                crate::ir::expr::Op::Bin(_, x, y) => {
                    self.depends_on_pins(x) || self.depends_on_pins(y)
                }
                crate::ir::expr::Op::Un(_, x)
                | crate::ir::expr::Op::Zext(x)
                | crate::ir::expr::Op::Sext(x)
                | crate::ir::expr::Op::Trunc(x) => self.depends_on_pins(x),
                crate::ir::expr::Op::Select(c, x, y) => {
                    self.depends_on_pins(c) || self.depends_on_pins(x) || self.depends_on_pins(y)
                }
            }
        };
        self.pin_dep.insert(r, out);
        out
    }

    /// Model an SSE data move, returning whether it was handled. Only pure movement is modelled, in 8- and 16-byte units: an XMM register is tracked as two 64-bit halves, so anything narrower or anything doing arithmetic returns `false` and is boxed as before.
    fn try_sse_move(&mut self, inst: &Instruction, mnemonic: Mnemonic, site: u64) -> bool {
        use crate::vm::state::XmmHalf::{High, Low};
        use Mnemonic as M;

        // Width of the move, or `None` if not a form we model.
        let bytes = match mnemonic {
            M::Movups | M::Movaps | M::Movdqu | M::Movdqa | M::Movupd | M::Movapd => 16,
            M::Movsd | M::Movq | M::Movlps | M::Movlpd => 8,
            // Zeroing idiom, but only in the same-register form: otherwise it is a
            // real xor of two different values and we do not model it.
            M::Xorps | M::Xorpd | M::Pxor => {
                if inst.op_count() == 2
                    && inst.op_kind(0) == OpKind::Register
                    && inst.op_kind(1) == OpKind::Register
                    && inst.op_register(0) == inst.op_register(1)
                {
                    if let Some(d) = xmm_index(inst.op_register(0)) {
                        let zero = self.arena.constant(0, Width::W64);
                        self.state.set_xmm(d, Low, zero);
                        self.state.set_xmm(d, High, zero);
                        return true;
                    }
                }
                return false;
            }
            _ => return false,
        };

        if inst.op_count() != 2 {
            return false;
        }

        match (inst.op_kind(0), inst.op_kind(1)) {
            // xmm <- memory
            (OpKind::Register, OpKind::Memory) => {
                let Some(d) = xmm_index(inst.op_register(0)) else {
                    return false;
                };
                let addr = self.mem_address(inst);
                let lo = self.load(addr, Width::W64, site);
                self.state.set_xmm(d, Low, lo);
                if bytes == 16 {
                    let eight = self.arena.constant(8, Width::W64);
                    let hi_addr = self.arena.bin(BinOp::Add, addr, eight);
                    let hi = self.load(hi_addr, Width::W64, site);
                    self.state.set_xmm(d, High, hi);
                } else {
                    // An 8-byte load into an XMM register zeroes the upper half.
                    let zero = self.arena.constant(0, Width::W64);
                    self.state.set_xmm(d, High, zero);
                }
                true
            }
            // memory <- xmm
            (OpKind::Memory, OpKind::Register) => {
                let Some(s) = xmm_index(inst.op_register(1)) else {
                    return false;
                };
                // Unknown source: the destination bytes must not keep their old
                // contents, but we have nothing true to say about them either.
                let (Some(lo), hi) = (self.state.xmm(s, Low), self.state.xmm(s, High)) else {
                    let addr = self.mem_address(inst);
                    self.forget_memory(addr, bytes);
                    return true;
                };
                if bytes == 16 && hi.is_none() {
                    let addr = self.mem_address(inst);
                    self.forget_memory(addr, bytes);
                    return true;
                }
                let addr = self.mem_address(inst);
                self.store(addr, lo, Width::W64, site);
                if bytes == 16 {
                    let eight = self.arena.constant(8, Width::W64);
                    let hi_addr = self.arena.bin(BinOp::Add, addr, eight);
                    self.store(hi_addr, hi.unwrap(), Width::W64, site);
                }
                true
            }
            // xmm <- xmm
            (OpKind::Register, OpKind::Register) => {
                let (Some(d), Some(s)) = (
                    xmm_index(inst.op_register(0)),
                    xmm_index(inst.op_register(1)),
                ) else {
                    return false;
                };
                match self.state.xmm(s, Low) {
                    Some(lo) => self.state.set_xmm(d, Low, lo),
                    None => self.state.forget_xmm(d),
                }
                if bytes == 16 {
                    match self.state.xmm(s, High) {
                        Some(hi) => self.state.set_xmm(d, High, hi),
                        // Low half known, high half not: drop the whole register
                        // rather than leave a stale high half behind.
                        None => self.state.forget_xmm(d),
                    }
                } else {
                    let zero = self.arena.constant(0, Width::W64);
                    self.state.set_xmm(d, High, zero);
                }
                true
            }
            _ => false,
        }
    }

    /// Make the destination and flag effects of a boxed instruction opaque.
    fn boxed_uses(&mut self, inst: &Instruction) -> Vec<(Register, Ref)> {
        let op_code = inst.op_code();
        let mut encoded: Vec<Register> = Vec::new();
        for i in 0..inst.op_count() {
            if inst.op_kind(i) != OpKind::Register {
                continue;
            }
            // Rewritable only if the operand is a register *field* in the encoding.
            let rewritable = (i as u32) < op_code.op_count()
                && matches!(
                    op_code.op_kind(i as u32),
                    OpCodeOperandKind::r8_reg
                        | OpCodeOperandKind::r8_opcode
                        | OpCodeOperandKind::r16_reg
                        | OpCodeOperandKind::r16_reg_mem
                        | OpCodeOperandKind::r16_rm
                        | OpCodeOperandKind::r16_opcode
                        | OpCodeOperandKind::r32_reg
                        | OpCodeOperandKind::r32_reg_mem
                        | OpCodeOperandKind::r32_rm
                        | OpCodeOperandKind::r32_opcode
                        | OpCodeOperandKind::r32_vvvv
                        | OpCodeOperandKind::r64_reg
                        | OpCodeOperandKind::r64_reg_mem
                        | OpCodeOperandKind::r64_rm
                        | OpCodeOperandKind::r64_opcode
                        | OpCodeOperandKind::r64_vvvv
                        | OpCodeOperandKind::r8_or_mem
                        | OpCodeOperandKind::r16_or_mem
                        | OpCodeOperandKind::r32_or_mem
                        | OpCodeOperandKind::r64_or_mem
                );
            if rewritable {
                encoded.push(inst.op_register(i));
            }
        }
        let mut f = InstructionInfoFactory::new();
        let info = f.info(inst);
        let mut out: Vec<(Register, Ref)> = Vec::new();
        for u in info.used_registers() {
            let reads = matches!(
                u.access(),
                OpAccess::Read | OpAccess::ReadWrite | OpAccess::CondRead | OpAccess::ReadCondWrite
            );
            if !reads {
                continue;
            }
            let r = u.register();
            // Not values this emitter moves, and RIP is meaningless once relocated.
            if r == Register::None || r == Register::RIP {
                continue;
            }
            if encoded.contains(&r) {
                continue;
            }
            // Only registers the allocator manages can be re-homed. `decode_reg`
            // returning `None` means a segment/control/debug register, which
            // `read_reg` would answer with an `xreg` opaque; no value to move.
            if decode_reg(r).is_none() {
                continue;
            }
            if out.iter().any(|(existing, _)| *existing == r) {
                continue;
            }
            let v = self.read_reg(r);
            out.push((r, v));
        }
        out
    }

    fn apply_boxed_effects(
        &mut self,
        inst: &Instruction,
        mnemonic: Mnemonic,
        reemitted_mem: bool,
    ) -> Vec<(Register, Ref)> {
        use Mnemonic as M;

        // Registers this instruction leaves a value in, for `Event::Boxed::defs`.
        // Only the fixed-register hardware reads report here: the generic case below
        // clobbers a register named by the encoding, which `emit_boxed` re-emits
        // verbatim, so its destination is already whatever the original named.
        let mut defs: Vec<(Register, Ref)> = Vec::new();

        // Instructions with well-known register effects that the VM's own
        // dispatch never depends on, but whose destinations must not keep their
        // previous contents.
        match mnemonic {
            M::Cpuid => {
                for r in [Register::EAX, Register::EBX, Register::ECX, Register::EDX] {
                    let v = self.arena.opaque("cpuid", Width::W64);
                    self.write_reg(r, v);
                    defs.push((r, v));
                }
                return defs;
            }
            M::Rdtsc | M::Rdtscp => {
                for r in [Register::EAX, Register::EDX] {
                    let v = self.arena.opaque("rdtsc", Width::W64);
                    self.write_reg(r, v);
                    defs.push((r, v));
                }
                if mnemonic == M::Rdtscp {
                    let v = self.arena.opaque("rdtsc", Width::W64);
                    self.write_reg(Register::ECX, v);
                    defs.push((Register::ECX, v));
                }
                return defs;
            }
            M::Rdmsr => {
                for r in [Register::EAX, Register::EDX] {
                    let v = self.arena.opaque("rdmsr", Width::W64);
                    self.write_reg(r, v);
                    defs.push((r, v));
                }
                return defs;
            }
            M::Wrmsr
            | M::Invlpg
            | M::Wbinvd
            | M::Mfence
            | M::Lfence
            | M::Sfence
            | M::Vmxoff
            | M::Vmlaunch
            | M::Vmresume => return defs,
            _ => {}
        }

        // Generic case: clobber the first operand if it is written.
        if inst.op_count() > 0 {
            match inst.op_kind(0) {
                OpKind::Register => {
                    let reg = inst.op_register(0);
                    if decode_reg(reg).is_some() {
                        let v = self.arena.opaque("boxed", Width::W64);
                        self.write_reg(reg, v);
                    } else if let Some(x) = xmm_index(reg) {
                        // We model XMM moves, so a boxed instruction writing one has
                        // to invalidate what we think it holds. Without this a later
                        // move would store a stale value that the real instruction
                        // has already overwritten.
                        self.state.forget_xmm(x);
                    } else if reg.is_ymm() || reg.is_zmm() {
                        // The low 16 bytes alias XMM, which we do model.
                        self.state.forget_xmm(reg.number() as u8);
                    }
                }
                OpKind::Memory => {
                    let addr = self.mem_address(inst);
                    let bytes = inst.memory_size().size() as u32;
                    if reemitted_mem {
                        // Codegen re-emits this write from the instruction's own
                        // encoding, so forget the bytes rather than inventing a
                        // value for them. Uses the raw byte count: a `Width` cannot
                        // describe a 16-byte SSE operand.
                        self.forget_memory(addr, bytes);
                    } else {
                        let w = Width::from_bytes(bytes).unwrap_or(Width::W64);
                        let v = self.arena.opaque("boxed", w);
                        self.store(addr, v, w, inst.ip());
                    }
                }
                _ => {}
            }
        }

        // Boxed instructions that touch flags: mark them unknown rather than
        // guessing, so a later JCC on them is reported as unresolved instead of
        // silently taking a wrong edge.
        if inst.rflags_modified() != 0 {
            self.clobber_arith_flags();
        }

        defs
    }

    fn clobber_cf_of(&mut self) {
        let u = self.arena.opaque("cf", Width::W8);
        self.state.set_flag(Flag::Cf, u);
        let u = self.arena.opaque("of", Width::W8);
        self.state.set_flag(Flag::Of, u);
    }

    fn clobber_arith_flags(&mut self) {
        self.clobber_cf_of();
        for f in [Flag::Zf, Flag::Sf, Flag::Pf, Flag::Af] {
            let u = self.arena.opaque("flag", Width::W8);
            self.state.set_flag(f, u);
        }
    }

    /// Argument registers explicitly written during this run. Untouched entry values
    /// are retained when marked as written because the callee's arity is unknown.
    fn call_arguments(&self) -> Vec<(Reg, Ref)> {
        ARG_REGS
            .iter()
            .filter(|r| self.written_args.contains(r))
            .map(|r| (*r, self.state.reg(*r)))
            .collect()
    }

    /// Clobber the volatile registers a call destroys, returning the `call_ret`
    /// opaque left in RAX so the caller can record it on the event.
    fn clobber_call(&mut self) -> Option<Ref> {
        let mut ret = None;
        for r in [
            Reg::Rax,
            Reg::Rcx,
            Reg::Rdx,
            Reg::R8,
            Reg::R9,
            Reg::R10,
            Reg::R11,
        ] {
            // Only RAX carries the callee's result.
            let tag = if r == Reg::Rax {
                "call_ret"
            } else {
                "clobbered"
            };
            let u = self.arena.opaque(tag, Width::W64);
            if r == Reg::Rax {
                ret = Some(u);
            }
            self.state.set_reg(r, u);
        }
        // Any setup before this call belonged to it, not to the next one.
        self.written_args.clear();
        self.clobber_arith_flags();
        ret
    }

    /// Run until evaluation stops or the budget runs out.
    pub fn run(&mut self, start: u64, budget: usize) -> Stop {
        let mut ip = start;
        for _ in 0..budget {
            if self.pe.section_for_va(ip).is_none() {
                return Stop::OutOfImage { site: ip };
            }
            // Runaway DAG growth usually means constant folding has stopped working
            // (typically because a value the VM relies on never became concrete).
            // Continuing from here costs time without producing a usable result, and
            // the expressions get large enough to make every subsequent
            // simplification pass slow.
            if self.arena.len() > self.node_limit {
                return Stop::Diverged {
                    site: ip,
                    nodes: self.arena.len(),
                };
            }
            match self.step(ip) {
                Step::Next(n) => ip = n,
                Step::Stopped(s) => return s,
            }
        }
        Stop::Budget { site: ip }
    }

    pub fn discover_vip_slot(&mut self, start: u64, budget: usize) -> Option<u64> {
        const SAMPLE_EVERY: usize = 16;
        const WINDOW: u64 = 0x200;
        let mut seen: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut ip = start;
        for i in 0..budget {
            if self.pe.section_for_va(ip).is_none() {
                break;
            }
            if i % SAMPLE_EVERY == 0 {
                if let Some(base) = self.arena.as_const(self.state.reg(Reg::Rbp)) {
                    for off in (0..WINDOW).step_by(8) {
                        if let Some(v) = self.read_const_qword(base.wrapping_add(off)) {
                            if self.in_vm(v) && Some(v) != self.unwind_slot {
                                seen.entry(off).or_default().insert(v);
                            }
                        }
                    }
                }
            }
            match self.step(ip) {
                Step::Next(n) => ip = n,
                Step::Stopped(_) => break,
            }
        }
        pick_vip_slot(&seen)
    }
}

/// Whether the instruction has an FS/GS-relative memory operand. Only these two matter.
fn seg_relative(inst: &Instruction) -> bool {
    matches!(inst.segment_prefix(), Register::FS | Register::GS)
        && (0..inst.op_count()).any(|i| inst.op_kind(i) == OpKind::Memory)
}

/// Whether an unmodelled instruction can be treated as a boxed instruction. The VM boxes anything outside the AMD64 subset it models, so in principle everything is boxable.
fn is_boxable(m: Mnemonic) -> bool {
    use Mnemonic as M;
    !matches!(
        m,
        // Control flow that would take us somewhere unknown.
        M::Iret
            | M::Iretd
            | M::Iretq
            | M::Sysenter
            | M::Sysexit
            | M::Sysexitq
            | M::Syscall
            | M::Sysret
            | M::Sysretq
            | M::Int
            | M::Int3
            | M::Into
            | M::Ud0
            | M::Ud1
            | M::Ud2
            | M::Hlt
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionCode {
    O,
    No,
    B,
    Ae,
    E,
    Ne,
    Be,
    A,
    S,
    Ns,
    P,
    Np,
    L,
    Ge,
    Le,
    G,
}

fn jcc_cc(m: Mnemonic) -> Option<ConditionCode> {
    use ConditionCode as C;
    use Mnemonic as M;
    Some(match m {
        M::Jo => C::O,
        M::Jno => C::No,
        M::Jb => C::B,
        M::Jae => C::Ae,
        M::Je => C::E,
        M::Jne => C::Ne,
        M::Jbe => C::Be,
        M::Ja => C::A,
        M::Js => C::S,
        M::Jns => C::Ns,
        M::Jp => C::P,
        M::Jnp => C::Np,
        M::Jl => C::L,
        M::Jge => C::Ge,
        M::Jle => C::Le,
        M::Jg => C::G,
        _ => return None,
    })
}

fn setcc_cc(m: Mnemonic) -> Option<ConditionCode> {
    use ConditionCode as C;
    use Mnemonic as M;
    Some(match m {
        M::Seto => C::O,
        M::Setno => C::No,
        M::Setb => C::B,
        M::Setae => C::Ae,
        M::Sete => C::E,
        M::Setne => C::Ne,
        M::Setbe => C::Be,
        M::Seta => C::A,
        M::Sets => C::S,
        M::Setns => C::Ns,
        M::Setp => C::P,
        M::Setnp => C::Np,
        M::Setl => C::L,
        M::Setge => C::Ge,
        M::Setle => C::Le,
        M::Setg => C::G,
        _ => return None,
    })
}

fn cmov_cc(m: Mnemonic) -> Option<ConditionCode> {
    use ConditionCode as C;
    use Mnemonic as M;
    Some(match m {
        M::Cmovo => C::O,
        M::Cmovno => C::No,
        M::Cmovb => C::B,
        M::Cmovae => C::Ae,
        M::Cmove => C::E,
        M::Cmovne => C::Ne,
        M::Cmovbe => C::Be,
        M::Cmova => C::A,
        M::Cmovs => C::S,
        M::Cmovns => C::Ns,
        M::Cmovp => C::P,
        M::Cmovnp => C::Np,
        M::Cmovl => C::L,
        M::Cmovge => C::Ge,
        M::Cmovle => C::Le,
        M::Cmovg => C::G,
        _ => return None,
    })
}

fn acc_for_width(w: Width) -> Register {
    match w {
        Width::W8 => Register::AL,
        Width::W16 => Register::AX,
        Width::W32 => Register::EAX,
        Width::W64 => Register::RAX,
    }
}

fn dx_for_width(w: Width) -> Register {
    match w {
        Width::W8 => Register::AH,
        Width::W16 => Register::DX,
        Width::W32 => Register::EDX,
        Width::W64 => Register::RDX,
    }
}

/// Map an iced register to (64-bit base, byte offset, width). Index of an XMM register, or `None` if `reg` is not one.
fn xmm_index(reg: Register) -> Option<u8> {
    if reg.is_xmm() {
        Some(reg.number() as u8)
    } else {
        None
    }
}

/// Split an `iced` register into the 64-bit register it lives in, its byte offset
/// within it, and its width.
pub(crate) fn decode_reg(reg: Register) -> Option<(Reg, u32, Width)> {
    use Register as R;
    let r = match reg {
        R::RAX | R::EAX | R::AX | R::AL => Reg::Rax,
        R::RCX | R::ECX | R::CX | R::CL => Reg::Rcx,
        R::RDX | R::EDX | R::DX | R::DL => Reg::Rdx,
        R::RBX | R::EBX | R::BX | R::BL => Reg::Rbx,
        R::RSP | R::ESP | R::SP | R::SPL => Reg::Rsp,
        R::RBP | R::EBP | R::BP | R::BPL => Reg::Rbp,
        R::RSI | R::ESI | R::SI | R::SIL => Reg::Rsi,
        R::RDI | R::EDI | R::DI | R::DIL => Reg::Rdi,
        R::R8 | R::R8D | R::R8W | R::R8L => Reg::R8,
        R::R9 | R::R9D | R::R9W | R::R9L => Reg::R9,
        R::R10 | R::R10D | R::R10W | R::R10L => Reg::R10,
        R::R11 | R::R11D | R::R11W | R::R11L => Reg::R11,
        R::R12 | R::R12D | R::R12W | R::R12L => Reg::R12,
        R::R13 | R::R13D | R::R13W | R::R13L => Reg::R13,
        R::R14 | R::R14D | R::R14W | R::R14L => Reg::R14,
        R::R15 | R::R15D | R::R15W | R::R15L => Reg::R15,
        R::AH => return Some((Reg::Rax, 1, Width::W8)),
        R::CH => return Some((Reg::Rcx, 1, Width::W8)),
        R::DH => return Some((Reg::Rdx, 1, Width::W8)),
        R::BH => return Some((Reg::Rbx, 1, Width::W8)),
        _ => return None,
    };
    let w = Width::from_bytes(reg.size() as u32)?;
    Some((r, 0, w))
}

/// Whether a guest-image slot's contents confirm that it holds `reg`. Used to score a candidate context base: the slots are the only evidence, so this predicate decides where the guest register image is believed to be.
fn is_iat_tail_call(slot: u64, import_slots: &std::collections::HashSet<u64>) -> bool {
    import_slots.contains(&slot)
}

fn slot_confirms_register(op: &Op, reg: Reg) -> bool {
    match op {
        Op::InitReg(r) => *r == reg,
        Op::Param(_, r) => *r == reg,
        _ => false,
    }
}

/// Choose the VIP slot from per-offset observations of distinct `.tvm0` values. Split out from [`Emulator::discover_vip_slot`] so the decision can be tested without a binary: the probe needs a loaded PE and a running interpreter, the choice needs neither.
fn pick_vip_slot(seen: &HashMap<u64, HashSet<u64>>) -> Option<u64> {
    seen.iter()
        .max_by_key(|(off, vals)| (vals.len(), std::cmp::Reverse(**off)))
        .filter(|(_, vals)| vals.len() > 1)
        .map(|(off, _)| *off)
}

#[cfg(test)]
mod vip_slot_tests {
    use super::*;

    fn obs(pairs: &[(u64, &[u64])]) -> HashMap<u64, HashSet<u64>> {
        pairs
            .iter()
            .map(|(o, vs)| (*o, vs.iter().copied().collect()))
            .collect()
    }

    /// The slot that moves most wins, which is what makes the offset derivable
    /// instead of hardcoded.
    #[test]
    fn the_slot_that_moves_most_is_the_vip() {
        let many: Vec<u64> = (0..130).collect();
        let few: Vec<u64> = (0..7).collect();
        let seen = obs(&[
            (0x60, &many),
            (0x90, &few),
            (0x98, &few),
            (0x8, &[1]),
            (0x20, &[2]),
            (0x30, &[3]),
        ]);
        assert_eq!(pick_vip_slot(&seen), Some(0x60));
    }

    /// A constant slot is not a position.
    ///
    /// Accepting one would be worse than returning nothing: it looks like a valid
    /// vip, so identity silently degenerates to the handler alone rather than
    /// falling back to the handler-only path on purpose.
    #[test]
    fn a_slot_that_never_moves_is_not_a_vip() {
        assert_eq!(pick_vip_slot(&obs(&[(0x60, &[0x1400a3c2])])), None);
        assert_eq!(pick_vip_slot(&obs(&[])), None);
        // Several slots, none of which move.
        assert_eq!(
            pick_vip_slot(&obs(&[(0x8, &[1]), (0x60, &[2]), (0x90, &[3])])),
            None
        );
    }

    /// A tie must resolve the same way every run.
    #[test]
    fn a_tie_resolves_to_the_lower_offset() {
        let a: Vec<u64> = (0..20).collect();
        for _ in 0..64 {
            let seen = obs(&[(0x98, &a), (0x60, &a), (0x90, &a)]);
            assert_eq!(
                pick_vip_slot(&seen),
                Some(0x60),
                "ties must not depend on hash order"
            );
        }
    }
}

#[cfg(test)]
mod ctx_tests {
    use super::*;
    use crate::ir::expr::{Arena, BlockRef};

    /// A seeded slot still confirms its register, so cutting a block does not erase
    /// the evidence that locates the guest context.
    #[test]
    fn a_seeded_slot_still_confirms_its_register() {
        let mut a = Arena::new();

        // Untouched: the register's entry value.
        let init = a.init_reg(Reg::Rcx);
        assert!(slot_confirms_register(a.op(init), Reg::Rcx));

        // Cut: an SSA parameter for the same register, which only gets written here
        // once the image has already been found at this base.
        let p = a.param(BlockRef(3), Reg::Rcx);
        assert!(
            slot_confirms_register(a.op(p), Reg::Rcx),
            "a cut block must not erase its own context base"
        );

        // Both forms must still discriminate on the register, or the layout probe
        // cannot tell the candidate slot orderings apart.
        let other_init = a.init_reg(Reg::Rdx);
        let other_param = a.param(BlockRef(3), Reg::Rdx);
        assert!(!slot_confirms_register(a.op(other_init), Reg::Rcx));
        assert!(!slot_confirms_register(a.op(other_param), Reg::Rcx));

        // Anything else is not evidence either way.
        let c = a.constant(0x140030ab0, Width::W64);
        assert!(!slot_confirms_register(a.op(c), Reg::Rcx));
        let sym = a.opaque("call_ret", Width::W64);
        assert!(!slot_confirms_register(a.op(sym), Reg::Rcx));
    }
}

#[cfg(test)]
mod tail_call_tests {
    use super::*;
    use std::collections::HashSet;

    /// Only an exact IAT slot is a tail call.
    #[test]
    fn only_an_exact_iat_slot_is_a_tail_call() {
        let slots: HashSet<u64> = [0x14002a000, 0x14002a008, 0x14002a168]
            .into_iter()
            .collect();

        assert!(
            is_iat_tail_call(0x14002a168, &slots),
            "a real slot is a tail call"
        );
        assert!(is_iat_tail_call(0x14002a000, &slots));

        assert!(
            !is_iat_tail_call(0x14002a010, &slots),
            "a gap in the IAT is not a slot"
        );
        // Misaligned into the middle of a real slot.
        assert!(!is_iat_tail_call(0x14002a16c, &slots));
        // A VM dispatch table entry, which uses the identical instruction form.
        assert!(
            !is_iat_tail_call(0x14002d038, &slots),
            "a dispatch table is not the IAT"
        );
        // A handler address in the VM section.
        assert!(!is_iat_tail_call(0x1401e1897, &slots));

        // An empty set means no imports were parsed; nothing may be a tail call,
        // because guessing would truncate every function that dispatches this way.
        assert!(!is_iat_tail_call(0x14002a168, &HashSet::new()));
    }
}

#[cfg(test)]
mod segment_tests {
    use super::*;
    use iced_x86::{Decoder, DecoderOptions};

    fn decode(bytes: &[u8]) -> Instruction {
        let mut dec = Decoder::with_ip(64, bytes, 0x1400a4c4d, DecoderOptions::NONE);
        dec.decode()
    }

    /// FS/GS-relative accesses are segment-relative; nothing else is.
    #[test]
    fn only_fs_and_gs_are_segment_relative() {
        let gs = decode(&[0x65, 0x8a, 0x0c, 0x25, 0x84, 0x01, 0x00, 0x00]);
        assert_eq!(gs.mnemonic(), Mnemonic::Mov);
        assert!(seg_relative(&gs), "a gs-prefixed load is segment-relative");

        // `mov eax, fs:[18h]`.
        let fs = decode(&[0x64, 0x8b, 0x04, 0x25, 0x18, 0x00, 0x00, 0x00]);
        assert!(seg_relative(&fs), "an fs-prefixed load is segment-relative");

        // The same load with no prefix: an ordinary absolute access, which models
        // fine and must not be diverted to boxing.
        let plain = decode(&[0x8a, 0x0c, 0x25, 0x84, 0x01, 0x00, 0x00]);
        assert!(!seg_relative(&plain), "no prefix is not segment-relative");

        // A DS-prefixed access. In long mode DS is forced to zero, so the prefix
        // contributes nothing; treating it as segment-relative would box addresses
        // that resolve perfectly well.
        let ds = decode(&[0x3e, 0x8b, 0x04, 0x25, 0x18, 0x00, 0x00, 0x00]);
        assert!(!seg_relative(&ds), "ds contributes no base in long mode");

        // A register-only instruction carries no memory operand to be relative to.
        let reg_only = decode(&[0x48, 0x89, 0xc8]);
        assert!(!seg_relative(&reg_only), "no memory operand");
    }
}

/// VMX instructions that fall through can be preserved as boxed instructions.
#[cfg(test)]
mod vmx_boxing_tests {
    use super::*;

    /// `vmxoff` and `vmlaunch` appear as guest instructions, so they must be boxable.
    #[test]
    fn vmx_fallthrough_instructions_are_boxable() {
        assert!(is_boxable(Mnemonic::Vmxoff), "0x1400a8654 needs this");
        assert!(is_boxable(Mnemonic::Vmlaunch), "0x1401175fa needs this");
        assert!(
            is_boxable(Mnemonic::Vmresume),
            "same family, same treatment"
        );
        assert!(is_boxable(Mnemonic::Vmxon), "was never excluded");
    }

    /// Boxing is still refused where continuing would be meaningless. Boxable means "replay the bytes and keep evaluating the next instruction". That is only sound when the instruction falls through to `next`.
    #[test]
    fn control_transfers_and_traps_are_still_not_boxable() {
        for m in [
            Mnemonic::Iretq,
            Mnemonic::Syscall,
            Mnemonic::Sysret,
            Mnemonic::Int3,
            Mnemonic::Ud2,
            Mnemonic::Hlt,
        ] {
            assert!(!is_boxable(m), "{m:?} does not fall through");
        }
    }
}

/// Whether `entry` is a routine that cannot return, by inspecting its body. Recognises the MSVC noreturn wrapper: straight-line setup, a call, and `int3` in the slot where the return would land.
fn callee_never_returns(pe: &PeFile, entry: u64) -> bool {
    let mut ip = entry;
    for _ in 0..NORETURN_SCAN_INSTRS {
        let Some(inst) = disasm::decode_at(pe, ip) else {
            return false;
        };
        let next = inst.next_ip();
        match inst.mnemonic() {
            // The shape being looked for: a call whose return slot is a trap.
            Mnemonic::Call => {
                return disasm::decode_at(pe, next).is_some_and(|i| i.mnemonic() == Mnemonic::Int3);
            }
            // Reached a trap without an intervening call: still cannot return.
            Mnemonic::Int3 | Mnemonic::Ud2 => return true,
            // A return path exists, or control leaves in a way this scan cannot
            // follow. Either way the straight-line assumption is broken.
            Mnemonic::Ret | Mnemonic::Jmp => return false,
            m if jcc_cc(m).is_some() => return false,
            _ => ip = next,
        }
    }
    false
}
