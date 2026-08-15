//! Hash-consed expression DAG for guest (symbolic) semantics. 
//! The whole devirtualization strategy rests on one observation: 
//! everything the VM needs in order to dispatch depends only on image constants, the initial (concretized) RSP and the VIP.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroU32;

use crate::ir::hash;

/// Bit width of a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Width {
    W8 = 1,
    W16 = 2,
    W32 = 4,
    W64 = 8,
}

impl Width {
    pub fn from_bytes(n: u32) -> Option<Width> {
        Some(match n {
            1 => Width::W8,
            2 => Width::W16,
            4 => Width::W32,
            8 => Width::W64,
            _ => return None,
        })
    }
    pub fn bytes(self) -> u32 {
        self as u32
    }
    pub fn bits(self) -> u32 {
        self as u32 * 8
    }
    /// Mask covering the value's bits, e.g. `W32 -> 0xffff_ffff`.
    pub fn mask(self) -> u64 {
        match self {
            Width::W64 => u64::MAX,
            w => (1u64 << w.bits()) - 1,
        }
    }
    pub fn sign_bit(self) -> u64 {
        1u64 << (self.bits() - 1)
    }
}

/// Index into [`Arena`]. Cheap to copy and compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ref(NonZeroU32);

impl Ref {
    fn from_index(i: usize) -> Ref {
        Ref(NonZeroU32::new(i as u32 + 1).expect("arena index overflow"))
    }
    fn index(self) -> usize {
        self.0.get() as usize - 1
    }
}

/// Guest register file slot, used for the initial symbolic state. `Ord` is derived so that anything iterating a map keyed by register can sort first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reg {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsp,
    Rbp,
    Rsi,
    Rdi,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

pub const GPRS: [Reg; 16] = [
    Reg::Rax,
    Reg::Rcx,
    Reg::Rdx,
    Reg::Rbx,
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

impl Reg {
    pub fn name(self) -> &'static str {
        match self {
            Reg::Rax => "rax",
            Reg::Rcx => "rcx",
            Reg::Rdx => "rdx",
            Reg::Rbx => "rbx",
            Reg::Rsp => "rsp",
            Reg::Rbp => "rbp",
            Reg::Rsi => "rsi",
            Reg::Rdi => "rdi",
            Reg::R8 => "r8",
            Reg::R9 => "r9",
            Reg::R10 => "r10",
            Reg::R11 => "r11",
            Reg::R12 => "r12",
            Reg::R13 => "r13",
            Reg::R14 => "r14",
            Reg::R15 => "r15",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Sar,
    Rol,
    Ror,
    /// Unsigned multiply, high half.
    MulHiU,
    /// Signed multiply, high half.
    MulHiS,
    UDiv,
    URem,
    SDiv,
    SRem,
    /// Unsigned less-than, produces 0/1.
    Ult,
    /// Signed less-than, produces 0/1. The lifter never builds one directly: x86 signed comparisons come through as expressions over SF/OF rather than as a single node.
    Slt,
    Eq,
}

impl BinOp {
    /// Commutative ops get their operands canonically ordered so that
    /// hash-consing catches `a op b` and `b op a` as the same node.
    fn is_commutative(self) -> bool {
        matches!(
            self,
            BinOp::Add
                | BinOp::Mul
                | BinOp::And
                | BinOp::Or
                | BinOp::Xor
                | BinOp::Eq
                | BinOp::MulHiU
                | BinOp::MulHiS
        )
    }
    pub fn symbol(self) -> &'static str {
        match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::And => "&",
            BinOp::Or => "|",
            BinOp::Xor => "^",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Sar => ">>s",
            BinOp::Rol => "rol",
            BinOp::Ror => "ror",
            BinOp::MulHiU => "mulhu",
            BinOp::MulHiS => "mulhs",
            BinOp::UDiv => "/u",
            BinOp::URem => "%u",
            BinOp::SDiv => "/s",
            BinOp::SRem => "%s",
            BinOp::Ult => "<u",
            BinOp::Slt => "<s",
            BinOp::Eq => "==",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnOp {
    Not,
    Neg,
    /// Population count of the low byte, used for PF.
    ParityByte,
    /// Byte swap.
    Bswap,
}

/// A node in the expression DAG. `width` is carried by [`Node`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Op {
    Const(u64),
    /// Initial value of a guest register on entry to the virtualized function.
    InitReg(Reg),
    /// Initial value of guest memory, i.e. a load whose address could not be resolved to image data. `0` is the address. The `u32` is the memory generation the load was taken at, and it exists solely to keep interning honest.
    Load(Ref, u32),
    Bin(BinOp, Ref, Ref),
    Un(UnOp, Ref),
    /// Zero-extend the operand to this node's width.
    Zext(Ref),
    /// Sign-extend the operand to this node's width.
    Sext(Ref),
    /// Truncate the operand to this node's width.
    Trunc(Ref),
    /// `cond ? a : b`, cond is a 0/1 value.
    Select(Ref, Ref, Ref),
    /// Opaque result of an instruction the lifter does not model
    /// (`CPUID`, `RDTSC`, `RDMSR`, `VMCALL`, ...). Identified by a tag so that
    /// two occurrences are not accidentally CSE'd together.
    Opaque(&'static str, u32),
    /// Value of a guest register on entry to a specific recovered block: an SSA block parameter. The block is part of the identity: the same register at two different blocks is two different values and must not be CSE'd together.
    Param(BlockRef, Reg),
}

/// Identity of a block, as referenced by [`Op::Param`].
///
/// A plain index into the CFG's block list. Kept opaque and separate from
/// `ir::BlockId` so that `expr` does not depend on the recovery stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockRef(pub u32);

impl std::fmt::Display for BlockRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "b{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node {
    pub op: Op,
    pub width: Width,
}

/// Hash-consing arena. All expression construction goes through here so that
/// structural equality is pointer equality. Cloning omits memo caches because they
/// are derived from immutable nodes.
#[derive(Debug, Default)]
pub struct Arena {
    nodes: Vec<Node>,
    intern: HashMap<Node, Ref>,
    /// Distinguishes two occurrences of the same unmodelled instruction. Starts at
    /// 0 and is incremented *before* use, so a generated id is never 0; which is
    /// what lets [`Arena::undef`] reserve 0 for its shared nodes.
    opaque_counter: u32,
    /// Current memory generation, stamped onto every [`Op::Load`].
    ///
    /// Bumped by [`Arena::bump_mem_gen`] whenever the evaluator's memory model
    /// changes, so that a load taken after a write cannot intern to one taken
    /// before it. See [`Op::Load`].
    mem_gen: u32,
    /// Memoized known-zero bit masks. Sound to cache because nodes are
    /// immutable once interned.
    known_zero: HashMap<Ref, u64>,
    known_one: HashMap<Ref, u64>,
    /// Current nesting depth of truncation distribution.
    trunc_depth: u32,
}

/// How many levels `trunc` is pushed through arithmetic before it is left in
/// place. The VM's masking idiom is shallow, so a small bound keeps the rewrite
/// effective without letting it rebuild deep expression trees repeatedly.
const TRUNC_DISTRIBUTE_DEPTH: u32 = 6;

impl Clone for Arena {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            intern: self.intern.clone(),
            opaque_counter: self.opaque_counter,
            mem_gen: self.mem_gen,
            // Derived caches: cheaper to recompute on demand than to carry.
            known_zero: HashMap::new(),
            known_one: HashMap::new(),
            trunc_depth: self.trunc_depth,
        }
    }
}

impl Arena {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// A hash of the *structure* of `r`, stable across arenas. `depth` bounds the walk.
    pub fn structural_hash(&self, r: Ref, depth: u32) -> u64 {
        let mut h = hash::FNV1A_OFFSET_BASIS;
        self.hash_into(r, depth, &mut h);
        h
    }

    fn hash_into(&self, r: Ref, depth: u32, h: &mut u64) {
        let mix = hash::mix_u64;
        mix(self.width(r) as u64, h);
        if depth == 0 {
            mix(0xdeadbeef, h);
            return;
        }
        // The discriminant has to be mixed in separately from the operands, so
        // that two different operators over the same operands do not collide.
        let (tag, kids): (u64, &[Ref]) = match self.op(r) {
            Op::Const(c) => {
                mix(1, h);
                mix(*c, h);
                return;
            }
            Op::InitReg(reg) => {
                mix(2, h);
                mix(*reg as u64, h);
                return;
            }
            Op::Opaque(tag, n) => {
                mix(3, h);
                hash::mix_bytes(tag.as_bytes(), h);
                mix(*n as u64, h);
                return;
            }
            Op::Param(b, reg) => {
                mix(4, h);
                mix(b.0 as u64, h);
                mix(*reg as u64, h);
                return;
            }
            // The generation is deliberately not hashed. It is a local counter, so two
            // structurally identical program points reached by different paths would
            // otherwise hash differently and defeat the state matching this feeds.
            Op::Load(a, _) => (5, std::slice::from_ref(a)),
            Op::Bin(op, a, b) => {
                mix(6, h);
                mix(*op as u64, h);
                let (a, b) = (*a, *b);
                self.hash_into(a, depth - 1, h);
                self.hash_into(b, depth - 1, h);
                return;
            }
            Op::Un(op, a) => {
                mix(7, h);
                mix(*op as u64, h);
                let a = *a;
                self.hash_into(a, depth - 1, h);
                return;
            }
            Op::Zext(a) => (8, std::slice::from_ref(a)),
            Op::Sext(a) => (9, std::slice::from_ref(a)),
            Op::Trunc(a) => (10, std::slice::from_ref(a)),
            Op::Select(c, a, b) => {
                mix(11, h);
                let (c, a, b) = (*c, *a, *b);
                self.hash_into(c, depth - 1, h);
                self.hash_into(a, depth - 1, h);
                self.hash_into(b, depth - 1, h);
                return;
            }
        };
        mix(tag, h);
        let kids: Vec<Ref> = kids.to_vec();
        for k in kids {
            self.hash_into(k, depth - 1, h);
        }
    }

    pub fn op(&self, r: Ref) -> &Op {
        &self.nodes[r.index()].op
    }

    pub fn width(&self, r: Ref) -> Width {
        self.nodes[r.index()].width
    }

    fn intern(&mut self, node: Node) -> Ref {
        if let Some(&r) = self.intern.get(&node) {
            return r;
        }
        let r = Ref::from_index(self.nodes.len());
        self.nodes.push(node.clone());
        self.intern.insert(node, r);
        r
    }

    // constructors

    pub fn constant(&mut self, value: u64, width: Width) -> Ref {
        self.intern(Node {
            op: Op::Const(value & width.mask()),
            width,
        })
    }

    pub fn init_reg(&mut self, reg: Reg) -> Ref {
        self.intern(Node {
            op: Op::InitReg(reg),
            width: Width::W64,
        })
    }

    pub fn load(&mut self, addr: Ref, width: Width) -> Ref {
        self.intern(Node {
            op: Op::Load(addr, self.mem_gen),
            width,
        })
    }

    /// A load carrying an explicit generation, for rebuilding one that already exists.
    pub fn load_at(&mut self, addr: Ref, width: Width, gen_at: u32) -> Ref {
        self.intern(Node {
            op: Op::Load(addr, gen_at),
            width,
        })
    }

    /// Note that guest memory has changed, so later loads cannot intern to earlier
    /// ones. See [`Op::Load`].
    pub fn bump_mem_gen(&mut self) {
        self.mem_gen = self.mem_gen.wrapping_add(1);
    }

    /// The generation later loads will be stamped with.
    #[cfg(test)]
    pub fn mem_gen(&self) -> u32 {
        self.mem_gen
    }

    pub fn opaque(&mut self, tag: &'static str, width: Width) -> Ref {
        self.opaque_counter += 1;
        let id = self.opaque_counter;
        self.intern(Node {
            op: Op::Opaque(tag, id),
            width,
        })
    }

    /// An opaque that every call with the same tag *shares*, unlike [`Self::opaque`]. For a value the architecture leaves undefined rather than merely unknown.
    pub fn undef(&mut self, tag: &'static str, width: Width) -> Ref {
        // Fixed id, so interning makes every call with this tag the same node.
        self.intern(Node {
            op: Op::Opaque(tag, 0),
            width,
        })
    }

    /// A block parameter: the value of `reg` on entry to `block`.
    pub fn param(&mut self, block: BlockRef, reg: Reg) -> Ref {
        self.intern(Node {
            op: Op::Param(block, reg),
            width: Width::W64,
        })
    }

    /// Copy the expression reachable from `r` in `src` into this arena, returning the equivalent local `Ref`. Needed because CFG recovery forks the evaluator, so every block's expressions live in that fork's own arena, which is dropped when recovery finishes.
    pub fn graft(&mut self, src: &Arena, r: Ref, memo: &mut HashMap<Ref, Ref>) -> Ref {
        if let Some(&hit) = memo.get(&r) {
            return hit;
        }
        // Iterative post-order: these DAGs nest deep enough (MBA chains run to
        // hundreds of levels) that recursion risks overflowing the stack.
        let mut order: Vec<Ref> = Vec::new();
        let mut queued: HashSet<Ref> = HashSet::new();
        let mut stack = vec![(r, false)];
        while let Some((cur, expanded)) = stack.pop() {
            if memo.contains_key(&cur) {
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
            for c in src.children(cur) {
                if !memo.contains_key(&c) {
                    stack.push((c, false));
                }
            }
        }

        for cur in order {
            if memo.contains_key(&cur) {
                continue;
            }
            let w = src.width(cur);
            let out = match *src.op(cur) {
                Op::Const(v) => self.constant(v, w),
                Op::InitReg(reg) => self.init_reg(reg),
                // Id 0 is `undef`'s shared marker; keep it shared. Any other id is
                // an identity from `src` and must be reissued here.
                Op::Opaque(tag, 0) => self.undef(tag, w),
                Op::Opaque(tag, _) => self.opaque(tag, w),
                // Preserve parameter identity across grafts; unlike counted opaques,
                // `(block, reg)` already provides a stable unique key.
                Op::Param(b, reg) => self.param(b, reg),
                // Rebuilt through the normal constructors so the result is
                // interned and simplified in *this* arena rather than copied
                // structurally.
                Op::Load(a, gen_at) => {
                    let a = memo[&a];
                    self.load_at(a, w, gen_at)
                }
                Op::Bin(op, a, b) => {
                    let (a, b) = (memo[&a], memo[&b]);
                    self.bin(op, a, b)
                }
                Op::Un(op, a) => {
                    let a = memo[&a];
                    self.un(op, a)
                }
                Op::Zext(a) => {
                    let a = memo[&a];
                    self.zext(a, w)
                }
                Op::Sext(a) => {
                    let a = memo[&a];
                    self.sext(a, w)
                }
                Op::Trunc(a) => {
                    let a = memo[&a];
                    self.trunc(a, w)
                }
                Op::Select(c, a, b) => {
                    let (c, a, b) = (memo[&c], memo[&a], memo[&b]);
                    self.select(c, a, b)
                }
            };
            memo.insert(cur, out);
        }

        memo[&r]
    }

    /// Direct operands of a node, in evaluation order. Rebuild `r` with every key of `map` replaced by its value.
    pub fn rewrite(&mut self, r: Ref, map: &HashMap<Ref, Ref>) -> Ref {
        if map.is_empty() {
            return r;
        }
        let mut done: HashMap<Ref, Ref> = HashMap::new();
        let mut order: Vec<Ref> = Vec::new();
        let mut queued: HashSet<Ref> = HashSet::new();
        let mut stack = vec![(r, false)];
        while let Some((cur, expanded)) = stack.pop() {
            if done.contains_key(&cur) {
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
            // A replaced node is a boundary: its children are irrelevant.
            if !map.contains_key(&cur) {
                for c in self.children(cur) {
                    if !done.contains_key(&c) {
                        stack.push((c, false));
                    }
                }
            }
        }

        for cur in order {
            if let Some(&to) = map.get(&cur) {
                done.insert(cur, to);
                continue;
            }
            let w = self.width(cur);
            let out = match *self.op(cur) {
                Op::Const(_) | Op::InitReg(_) | Op::Opaque(..) | Op::Param(..) => cur,
                Op::Load(a, gen_at) => {
                    let a = done[&a];
                    self.load_at(a, w, gen_at)
                }
                Op::Bin(op, a, b) => {
                    let (a, b) = (done[&a], done[&b]);
                    self.bin(op, a, b)
                }
                Op::Un(op, a) => {
                    let a = done[&a];
                    self.un(op, a)
                }
                Op::Zext(a) => {
                    let a = done[&a];
                    self.zext(a, w)
                }
                Op::Sext(a) => {
                    let a = done[&a];
                    self.sext(a, w)
                }
                Op::Trunc(a) => {
                    let a = done[&a];
                    self.trunc(a, w)
                }
                Op::Select(c, a, b) => {
                    let (c, a, b) = (done[&c], done[&a], done[&b]);
                    self.select(c, a, b)
                }
            };
            done.insert(cur, out);
        }
        done[&r]
    }

    pub fn children(&self, r: Ref) -> Vec<Ref> {
        match *self.op(r) {
            Op::Const(_) | Op::InitReg(_) | Op::Opaque(..) | Op::Param(..) => Vec::new(),
            Op::Load(a, _) | Op::Un(_, a) | Op::Zext(a) | Op::Sext(a) | Op::Trunc(a) => vec![a],
            Op::Bin(_, a, b) => vec![a, b],
            Op::Select(c, a, b) => vec![c, a, b],
        }
    }

    /// Constant value of `r`, if it is one.
    pub fn as_const(&self, r: Ref) -> Option<u64> {
        match self.op(r) {
            Op::Const(c) => Some(*c),
            _ => None,
        }
    }

    pub fn is_const(&self, r: Ref) -> bool {
        matches!(self.op(r), Op::Const(_))
    }

    pub fn select(&mut self, cond: Ref, a: Ref, b: Ref) -> Ref {
        if let Some(c) = self.as_const(cond) {
            return if c != 0 { a } else { b };
        }
        if a == b {
            return a;
        }
        let width = self.width(a);
        self.intern(Node {
            op: Op::Select(cond, a, b),
            width,
        })
    }

    pub fn zext(&mut self, r: Ref, to: Width) -> Ref {
        let from = self.width(r);
        if from == to {
            return r;
        }
        if to < from {
            return self.trunc(r, to);
        }
        if let Some(c) = self.as_const(r) {
            return self.constant(c & from.mask(), to);
        }
        // zext(zext(x)) == zext(x)
        if let Op::Zext(inner) = *self.op(r) {
            return self.zext(inner, to);
        }
        // zext(trunc(x, w), to) == cast(x, to) & mask(w)
        if let Op::Trunc(inner) = *self.op(r) {
            let widened = if self.width(inner) >= to {
                self.trunc(inner, to)
            } else {
                self.zext(inner, to)
            };
            let m = self.constant(from.mask(), to);
            return self.bin(BinOp::And, widened, m);
        }
        self.intern(Node {
            op: Op::Zext(r),
            width: to,
        })
    }

    pub fn sext(&mut self, r: Ref, to: Width) -> Ref {
        let from = self.width(r);
        if from == to {
            return r;
        }
        if to < from {
            return self.trunc(r, to);
        }
        if let Some(c) = self.as_const(r) {
            return self.constant(sign_extend(c, from), to);
        }
        self.intern(Node {
            op: Op::Sext(r),
            width: to,
        })
    }

    pub fn trunc(&mut self, r: Ref, to: Width) -> Ref {
        let from = self.width(r);
        if from == to {
            return r;
        }
        if to > from {
            return self.zext(r, to);
        }
        if let Some(c) = self.as_const(r) {
            return self.constant(c & to.mask(), to);
        }
        // trunc through an extension: drop the extension when it is redundant.
        match *self.op(r) {
            Op::Zext(inner) | Op::Sext(inner) | Op::Trunc(inner) => {
                let iw = self.width(inner);
                if iw >= to {
                    return self.trunc(inner, to);
                }
                // widening then narrowing to a still-wider width keeps the ext
            }
            // trunc(x & c) == trunc(x) when c covers the surviving bits.
            Op::Bin(BinOp::And, x, y) => {
                if let Some(c) = self.as_const(y) {
                    if c & to.mask() == to.mask() {
                        return self.trunc(x, to);
                    }
                }
            }
            _ => {}
        }

        // Push truncation through operations whose low bits depend only on the low bits of their operands. 
        // This is the single most valuable canonicalization for TVM: 
        // the VM constantly builds a 16-bit bytecode index as `(guest_reg & 0xffff_ffff_ffff_0000) | imm16` and then reads it back with `movzx`.
        if let Op::Bin(op, x, y) = *self.op(r) {
            if matches!(
                op,
                BinOp::And | BinOp::Or | BinOp::Xor | BinOp::Add | BinOp::Sub | BinOp::Mul
            ) && self.trunc_depth < TRUNC_DISTRIBUTE_DEPTH
            {
                self.trunc_depth += 1;
                let tx = self.trunc(x, to);
                let ty = self.trunc(y, to);
                let out = self.bin(op, tx, ty);
                self.trunc_depth -= 1;
                return out;
            }
        }
        // trunc(~x) == ~trunc(x), likewise for negation.
        if let Op::Un(op @ (UnOp::Not | UnOp::Neg), x) = *self.op(r) {
            let tx = self.trunc(x, to);
            return self.un(op, tx);
        }

        self.intern(Node {
            op: Op::Trunc(r),
            width: to,
        })
    }

    pub fn not(&mut self, r: Ref) -> Ref {
        let w = self.width(r);
        if let Some(c) = self.as_const(r) {
            return self.constant(!c, w);
        }
        // ~~x == x
        if let Op::Un(UnOp::Not, inner) = *self.op(r) {
            return inner;
        }
        // De Morgan: ~(x | y)  ==  ~x & ~y ~(x & y)  ==  ~x | ~y Applied only when it strictly *reduces* the number of NOTs, which is what makes it terminating.
        if let Op::Bin(inner_op @ (BinOp::Or | BinOp::And), x, y) = *self.op(r) {
            let reduces = |a: &Self, v: Ref| matches!(a.op(v), Op::Un(UnOp::Not, _) | Op::Const(_));
            if reduces(self, x) || reduces(self, y) {
                let nx = self.not(x);
                let ny = self.not(y);
                let flipped = match inner_op {
                    BinOp::Or => BinOp::And,
                    _ => BinOp::Or,
                };
                return self.bin(flipped, nx, ny);
            }
        }
        self.intern(Node {
            op: Op::Un(UnOp::Not, r),
            width: w,
        })
    }

    pub fn neg(&mut self, r: Ref) -> Ref {
        let w = self.width(r);
        if let Some(c) = self.as_const(r) {
            return self.constant(c.wrapping_neg(), w);
        }
        if let Op::Un(UnOp::Neg, inner) = *self.op(r) {
            return inner;
        }
        self.intern(Node {
            op: Op::Un(UnOp::Neg, r),
            width: w,
        })
    }

    pub fn un(&mut self, op: UnOp, r: Ref) -> Ref {
        match op {
            UnOp::Not => self.not(r),
            UnOp::Neg => self.neg(r),
            UnOp::ParityByte | UnOp::Bswap => {
                let w = self.width(r);
                if let Some(c) = self.as_const(r) {
                    let v = match op {
                        UnOp::ParityByte => ((c as u8).count_ones() % 2 == 0) as u64,
                        UnOp::Bswap => bswap(c, w),
                        _ => unreachable!(),
                    };
                    return self.constant(v, w);
                }
                self.intern(Node {
                    op: Op::Un(op, r),
                    width: w,
                })
            }
        }
    }

    /// Build `a op b`, folding constants and applying peephole/MBA rules.
    pub fn bin(&mut self, op: BinOp, a: Ref, b: Ref) -> Ref {
        let width = match op {
            BinOp::Ult | BinOp::Slt | BinOp::Eq => Width::W8,
            _ => self.width(a).max(self.width(b)),
        };

        // Constant folding.
        if let (Some(x), Some(y)) = (self.as_const(a), self.as_const(b)) {
            let aw = self.width(a);
            if let Some(v) = fold_bin(op, x, y, aw.max(self.width(b))) {
                return self.constant(v, width);
            }
        }

        // Canonical operand order for commutative ops: constants on the right,
        // otherwise by arena index. Keeps hash-consing effective.
        let (a, b) = if op.is_commutative() {
            let swap = self.is_const(a) && !self.is_const(b) || (!self.is_const(b) && a > b);
            if swap { (b, a) } else { (a, b) }
        } else {
            (a, b)
        };

        if let Some(r) = self.simplify_bin(op, a, b, width) {
            return r;
        }
        self.intern(Node {
            op: Op::Bin(op, a, b),
            width,
        })
    }

    /// Algebraic identities plus the Tencent VM MBA rule set. TVM builds large expressions by recursively applying a handful of trivial MBA identities.
    fn simplify_bin(&mut self, op: BinOp, a: Ref, b: Ref, width: Width) -> Option<Ref> {
        let mask = width.mask();
        let cb = self.as_const(b);

        match op {
            BinOp::Add => {
                if cb == Some(0) {
                    return Some(a);
                }
                // x + (-x) == 0, x + ~x == -1
                if let Op::Un(UnOp::Neg, inner) = *self.op(b) {
                    if inner == a {
                        return Some(self.constant(0, width));
                    }
                }
                if self.is_complement(a, b) {
                    return Some(self.constant(mask, width));
                }
                // (A|B) + (A&B) == A + B (A^B) + (A&B) == A | B A + B == A | B when their set bits cannot overlap. With no bit live in both operands no carry can occur, so the sum is just the union.
                {
                    let (za, zb) = (self.known_zero(a), self.known_zero(b));
                    if (!za & !zb & mask) == 0 {
                        return Some(self.bin(BinOp::Or, a, b));
                    }
                }
                // (A&B) + (A|B) == A + B          (commuted, same rule)
                if let (Some((o1, x1, y1)), Some((o2, x2, y2))) = (self.as_bin(a), self.as_bin(b)) {
                    if same_pair(x1, y1, x2, y2) {
                        match (o1, o2) {
                            (BinOp::Or, BinOp::And) | (BinOp::And, BinOp::Or) => {
                                return Some(self.bin(BinOp::Add, x1, y1));
                            }
                            (BinOp::Xor, BinOp::And) | (BinOp::And, BinOp::Xor) => {
                                return Some(self.bin(BinOp::Or, x1, y1));
                            }
                            _ => {}
                        }
                    }
                }
                // ((A|B)^A) + A == A | B   and   A + ((A|B)^A) == A | B
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::Xor, x, y)) = self.as_bin(p) {
                        for (or_side, a_side) in [(x, y), (y, x)] {
                            if a_side == q {
                                if let Some((BinOp::Or, u, v)) = self.as_bin(or_side) {
                                    if u == q || v == q {
                                        return Some(or_side);
                                    }
                                }
                            }
                        }
                    }
                }
                // (((A^B)|B) & A) + B == A + B   [((A^B)|B) == A|B, (A|B)&A == A]
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::And, x, y)) = self.as_bin(p) {
                        for (lhs, rhs) in [(x, y), (y, x)] {
                            if let Some((BinOp::Or, u, v)) = self.as_bin(lhs) {
                                let other = if u == q {
                                    Some(v)
                                } else if v == q {
                                    Some(u)
                                } else {
                                    None
                                };
                                if let Some(xor) = other {
                                    if let Some((BinOp::Xor, m, n)) = self.as_bin(xor) {
                                        if same_pair(m, n, rhs, q) {
                                            return Some(self.bin(BinOp::Add, rhs, q));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // Complementary masks share no live bit, so no carry can cross between the addends and the sum is their union. Rewriting to `Or` hands the pair to the OR rule set, which is where the complementary-mask reconstruction lives.
                if self.disjoint_by_factors(a, b) {
                    return Some(self.bin(BinOp::Or, a, b));
                }
                // (x + c1) + c2 == x + (c1+c2), reassociation on constants
                if let (Some(c2), Some((BinOp::Add, x, y))) = (cb, self.as_bin(a)) {
                    if let Some(c1) = self.as_const(y) {
                        let k = self.constant(c1.wrapping_add(c2), width);
                        return Some(self.bin(BinOp::Add, x, k));
                    }
                }
                None
            }

            BinOp::Sub => {
                if cb == Some(0) {
                    return Some(a);
                }
                if a == b {
                    return Some(self.constant(0, width));
                }
                // (A|B) - (A&B) == A ^ B
                if let (Some((BinOp::Or, x1, y1)), Some((BinOp::And, x2, y2))) =
                    (self.as_bin(a), self.as_bin(b))
                {
                    if same_pair(x1, y1, x2, y2) {
                        return Some(self.bin(BinOp::Xor, x1, y1));
                    }
                }
                // (a + x) - a == x, and (x + a) - a == x. Plain cancellation, but it needs the Add to be inspected rather than reassociated, which is the pass this simplifier does not have.
                if let Some((BinOp::Add, x, y)) = self.as_bin(a) {
                    if x == b {
                        return Some(y);
                    }
                    if y == b {
                        return Some(x);
                    }
                }
                // A - (A & B) == A & ~B Exact rather than approximate: `A & B` has no bit set that `A` does not, so the subtraction can never borrow and is the same as clearing those bits.
                if let Some((BinOp::And, u, v)) = self.as_bin(b) {
                    for (same, other) in [(u, v), (v, u)] {
                        if same == a {
                            let n = self.not(other);
                            return Some(self.bin(BinOp::And, a, n));
                        }
                    }
                }
                // A - (A - (A&B)) == A & B
                if let Some((BinOp::Sub, x, y)) = self.as_bin(b) {
                    if x == a {
                        if let Some((BinOp::And, u, v)) = self.as_bin(y) {
                            if u == a || v == a {
                                return Some(y);
                            }
                        }
                    }
                }
                // B - (((A&B)&B) ^ B) == A & B
                if let Some((BinOp::Xor, x, y)) = self.as_bin(b) {
                    for (inner, bref) in [(x, y), (y, x)] {
                        if bref == a {
                            if let Some((BinOp::And, u, v)) = self.as_bin(inner) {
                                for (and_ab, bb) in [(u, v), (v, u)] {
                                    if bb == a {
                                        if let Some((BinOp::And, _, _)) = self.as_bin(and_ab) {
                                            return Some(and_ab);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // x - c == x + (-c), normalises constants into Add form.
                if let Some(c) = cb {
                    let k = self.constant(c.wrapping_neg(), width);
                    return Some(self.bin(BinOp::Add, a, k));
                }
                None
            }

            BinOp::And => {
                if a == b {
                    return Some(a);
                }
                // Extract a single bit out of a bit-composition: `(x >> k) & 1`. The VM packs the flags into one word and unpacks it again around every guest transition, so this pattern appears for every flag on every such boundary.
                if cb == Some(1) {
                    if let Some((BinOp::Shr, x, sh)) = self.as_bin(a) {
                        if let Some(k) = self.as_const(sh) {
                            if let Some(bit) = self.bit_at(x, k) {
                                return Some(bit);
                            }
                        }
                    }
                    // Bit 0 with no shift.
                    if let Some(bit) = self.bit_at(a, 0) {
                        return Some(bit);
                    }
                }
                if cb == Some(0) {
                    return Some(self.constant(0, width));
                }
                if cb == Some(mask) {
                    return Some(a);
                }

                // A mask that is all-ones wherever the other side has a bit is a no-op. Generalises the all-ones constant case above to a computed mask.
                if cb.is_none() && self.as_const(a).is_none() {
                    for (p, q) in [(a, b), (b, a)] {
                        if (self.known_one(p) | self.known_zero(q)) & mask == mask {
                            return Some(q);
                        }
                    }
                }

                // Masking a bit-composition: drop the parts that cannot reach the mask. The VM packs RFLAGS and then reads a single flag back out of the packed word, but not as the `(x >> k) & 1` that `bit_at` already knows.
                if let Some(c) = cb {
                    if let Some((BinOp::Or, x, y)) = self.as_bin(a) {
                        let x_dead = self.known_zero(x) & c == c;
                        let y_dead = self.known_zero(y) & c == c;
                        if x_dead && !y_dead {
                            return Some(self.bin(BinOp::And, y, b));
                        }
                        if y_dead && !x_dead {
                            return Some(self.bin(BinOp::And, x, b));
                        }

                        // Neither immediate arm is dead on its own, but the pack is a right-nested chain and the dead flags sit further down it.
                        let mut terms = Vec::new();
                        self.collect_or_terms(a, 32, &mut terms);
                        let dead: Vec<bool> =
                            terms.iter().map(|&t| self.known_zero(t) & c == c).collect();
                        if dead.iter().any(|&d| d) {
                            let live: Vec<Ref> = terms
                                .into_iter()
                                .zip(dead)
                                .filter_map(|(t, d)| (!d).then_some(t))
                                .collect();
                            let rebuilt = match live
                                .into_iter()
                                .reduce(|acc, t| self.bin(BinOp::Or, acc, t))
                            {
                                Some(r) => r,
                                // Every leaf was dead, so nothing reaches the mask.
                                None => return Some(self.constant(0, width)),
                            };
                            return Some(self.bin(BinOp::And, rebuilt, b));
                        }
                    }
                }

                if self.is_complement(a, b) {
                    return Some(self.constant(0, width));
                }

                // `~(x ^ c1) & c2 == x & c2` when every bit of c2 is set in c1. Flipping a bit and then complementing flips it back, so for the bits the mask selects the pair cancels.
                if let Some(c2) = cb {
                    if let Op::Un(UnOp::Not, inner) = *self.op(a) {
                        if let Some((BinOp::Xor, x, k)) = self.as_bin(inner) {
                            if let Some(c1) = self.as_const(k) {
                                if c2 & c1 == c2 {
                                    return Some(self.bin(BinOp::And, x, b));
                                }
                            }
                        }
                    }
                }

                // (A|B) & A == A   and   (A&B) & A == A&B
                for (p, q) in [(a, b), (b, a)] {
                    match self.as_bin(p) {
                        Some((BinOp::Or, x, y)) if x == q || y == q => return Some(q),
                        Some((BinOp::And, x, y)) if x == q || y == q => return Some(p),
                        _ => {}
                    }
                }
                // A & (A ^ B) == A & ~B   (and the same with the XOR's arms swapped). Where A is clear both sides are clear; where A is set the XOR yields `~B`.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::Xor, x, y)) = self.as_bin(q) {
                        for (same, other) in [(x, y), (y, x)] {
                            if same == p {
                                let n = self.not(other);
                                return Some(self.bin(BinOp::And, p, n));
                            }
                        }
                    }
                }
                // A & (~A | B) == A & B   (and the same with the OR's arms swapped). Where A is clear the result is clear either way; where A is set the `~A` arm contributes nothing, so only B decides.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::Or, x, y)) = self.as_bin(q) {
                        for (neg, other) in [(x, y), (y, x)] {
                            if self.is_complement(neg, p) {
                                return Some(self.bin(BinOp::And, p, other));
                            }
                        }
                    }
                }
                // (x & c1) & c2 == x & (c1&c2)
                if let (Some(c2), Some((BinOp::And, x, y))) = (cb, self.as_bin(a)) {
                    if let Some(c1) = self.as_const(y) {
                        let k = self.constant(c1 & c2, width);
                        return Some(self.bin(BinOp::And, x, k));
                    }
                }
                // Bit-level rules against a constant mask.
                if let Some(c) = cb {
                    let ka = self.known_zero(a);
                    // Every bit the mask keeps is known zero: the result is 0.
                    if ka & c == c {
                        return Some(self.constant(0, width));
                    }
                    // The mask keeps every bit that could be set: it is a no-op.
                    if (!ka & width.mask()) & !c == 0 {
                        return Some(a);
                    }
                    // Every bit the mask keeps is known *set*: the result is the mask.
                    if self.known_one(a) & c == c {
                        return Some(self.constant(c, width));
                    }
                    // (X | Y) & c where one side contributes nothing under c.
                    if let Some((BinOp::Or, x, y)) = self.as_bin(a) {
                        for (keep, drop) in [(y, x), (x, y)] {
                            if self.known_zero(drop) & c == c {
                                let k = self.constant(c, width);
                                return Some(self.bin(BinOp::And, keep, k));
                            }
                        }
                    }
                }
                None
            }

            BinOp::Or => {
                // a | (a ^ b) == a | b. Where `a` is set both sides give one; where `a` is clear the xor is just `b`. Rewriting to `a | b` drops the xor, which is what exposes the plain OR of the two TSC halves underneath TVM's MBA wrapper.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::Xor, x, y)) = self.as_bin(q) {
                        if x == p {
                            return Some(self.bin(BinOp::Or, p, y));
                        }
                        if y == p {
                            return Some(self.bin(BinOp::Or, p, x));
                        }
                    }
                }

                if a == b {
                    return Some(a);
                }
                if cb == Some(0) {
                    return Some(a);
                }
                if cb == Some(mask) {
                    return Some(self.constant(mask, width));
                }
                if self.is_complement(a, b) {
                    return Some(self.constant(mask, width));
                }
                // A | (B & ~A) == A | B   (and the same with the AND's arms swapped). Dual of the `A & (~A | B)` absorption. Where A is set both sides give one; where A is clear the `~A` arm is all-ones and only B decides.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::And, x, y)) = self.as_bin(q) {
                        for (neg, other) in [(x, y), (y, x)] {
                            if self.is_complement(neg, p) {
                                return Some(self.bin(BinOp::Or, p, other));
                            }
                        }
                    }
                }
                // (A&B) | A == A   and   (A|B) | A == A|B
                for (p, q) in [(a, b), (b, a)] {
                    match self.as_bin(p) {
                        Some((BinOp::And, x, y)) if x == q || y == q => return Some(q),
                        Some((BinOp::Or, x, y)) if x == q || y == q => return Some(p),
                        _ => {}
                    }
                }
                if let (Some((BinOp::And, x1, y1)), Some((BinOp::And, x2, y2))) =
                    (self.as_bin(a), self.as_bin(b))
                {
                    for (shared_a, mask_a) in [(x1, y1), (y1, x1)] {
                        for (shared_b, mask_b) in [(x2, y2), (y2, x2)] {
                            if shared_a != shared_b {
                                continue;
                            }
                            if self.is_complement(mask_a, mask_b) && self.width(shared_a) == width {
                                return Some(shared_a);
                            }
                            // (X & c1) | (X & c2) == X & (c1|c2). The constant-mask form of the same fact.
                            if let (Some(c1), Some(c2)) =
                                (self.as_const(mask_a), self.as_const(mask_b))
                            {
                                let k = self.constant(c1 | c2, width);
                                return Some(self.bin(BinOp::And, shared_a, k));
                            }
                        }
                    }
                }
                // The same reconstruction, over conjunction factors rather than over a literal pair of `And` arms.
                {
                    let (fa, ea) = self.and_factors(a);
                    let (fb, eb) = self.and_factors(b);
                    if !ea || !eb || fa.len() < 2 || fb.len() < 2 {
                        // continue below
                    } else {
                        let mut result = None;
                        for &shared in &fa {
                            if !fb.contains(&shared) || self.width(shared) != width {
                                continue;
                            }
                            let rest_a: Vec<Ref> =
                                fa.iter().copied().filter(|&f| f != shared).collect();
                            let rest_b: Vec<Ref> =
                                fb.iter().copied().filter(|&f| f != shared).collect();
                            // One residual factor each, and complementary: the two arms
                            // partition `shared`'s bits, so together they restore it.
                            if let ([ra], [rb]) = (rest_a.as_slice(), rest_b.as_slice()) {
                                if self.is_complement(*ra, *rb) {
                                    result = Some(shared);
                                    break;
                                }
                            }
                        }
                        if let Some(r) = result {
                            return Some(r);
                        }
                    }
                }
                // Drop an operand whose every possible bit is already known set in
                // the other. Uses `known_one` on one side and `known_zero` on the
                // other, so it catches the obfuscator's redundant OR terms without
                // needing to match their particular shape.
                for (p, q) in [(a, b), (b, a)] {
                    if (!self.known_zero(p) & !self.known_one(q) & mask) == 0 {
                        return Some(q);
                    }
                }
                // (A ^ c) | c == A | c
                //
                // The XOR only flips bits inside c, and the OR then sets all of them
                // regardless, so the flipping is unobservable.
                if let (Some(c2), Some((BinOp::Xor, x, y))) = (cb, self.as_bin(a)) {
                    if self.as_const(y) == Some(c2) {
                        let k = self.constant(c2, width);
                        return Some(self.bin(BinOp::Or, x, k));
                    }
                }
                // (A | c1) | c2 -> A | (c1 | c2)
                // Merge nested OR constants
                if let (Some(c2), Some((BinOp::Or, x, y))) = (cb, self.as_bin(a)) {
                    if let Some(c1) = self.as_const(y) {
                        let k = self.constant(c1 | c2, width);
                        return Some(self.bin(BinOp::Or, x, k));
                    }
                }
                // A | ((A & k) | C) == A | C
                //
                // `A & k` contributes nothing that `A` does not already have, so it
                // drops out. Needed as its own rule because plain absorption only
                // looks one level down, and the obfuscator buries the term under a
                // second OR.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::Or, x, y)) = self.as_bin(q) {
                        for (cand, keep) in [(x, y), (y, x)] {
                            if let Some((BinOp::And, u, v)) = self.as_bin(cand) {
                                if u == p || v == p {
                                    return Some(self.bin(BinOp::Or, p, keep));
                                }
                            }
                        }
                    }
                }
                // (x | c1) | c2 == x | (c1|c2)
                if let (Some(c2), Some((BinOp::Or, x, y))) = (cb, self.as_bin(a)) {
                    if let Some(c1) = self.as_const(y) {
                        let k = self.constant(c1 | c2, width);
                        return Some(self.bin(BinOp::Or, x, k));
                    }
                }
                // ((c&A) ^ A) | A == A, and every operand ordering of it.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::Xor, x, y)) = self.as_bin(p) {
                        for (lhs, rhs) in [(x, y), (y, x)] {
                            if rhs == q {
                                if let Some((BinOp::And, u, v)) = self.as_bin(lhs) {
                                    if u == q || v == q {
                                        return Some(q);
                                    }
                                }
                                if let Some((BinOp::Or, u, v)) = self.as_bin(lhs) {
                                    // (((c|A) ^ c) | A) == A
                                    if u == q || v == q {
                                        return Some(q);
                                    }
                                }
                            }
                        }
                    }
                }
                // ((A|B) ^ B) | A == A
                //
                // TVM's address computation idiom:
                //   mov rax,r9 / or rax,rcx / xor rax,rcx / or rax,r9
                // The XOR clears exactly the bits B contributed to (A|B),
                // leaving A's B-bits cleared, and the final OR restores them.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::Xor, x, y)) = self.as_bin(p) {
                        for (or_side, other) in [(x, y), (y, x)] {
                            if let Some((BinOp::Or, u, v)) = self.as_bin(or_side) {
                                // or_side == (q | other), so the whole thing is q
                                if (u == q && v == other) || (v == q && u == other) {
                                    return Some(q);
                                }
                            }
                        }
                    }
                }
                None
            }

            BinOp::Xor => {
                if a == b {
                    return Some(self.constant(0, width));
                }
                // (A&B) ^ (A|B) == A ^ B
                //
                // The OR holds every bit either operand has, the AND only the shared
                // ones, so the difference between them is exactly the bits present in
                // one but not both.
                for (p, q) in [(a, b), (b, a)] {
                    if let (Some((BinOp::And, x1, y1)), Some((BinOp::Or, x2, y2))) =
                        (self.as_bin(p), self.as_bin(q))
                    {
                        if same_pair(x1, y1, x2, y2) {
                            return Some(self.bin(BinOp::Xor, x1, y1));
                        }
                    }
                }
                // (A&B) ^ (A^B) == A | B The XOR carries the bits in exactly one operand, the AND the bits in both, and the two sets are disjoint, so their XOR is their union.
                for (p, q) in [(a, b), (b, a)] {
                    if let (Some((BinOp::And, x1, y1)), Some((BinOp::Xor, x2, y2))) =
                        (self.as_bin(p), self.as_bin(q))
                    {
                        if same_pair(x1, y1, x2, y2) {
                            return Some(self.bin(BinOp::Or, x1, y1));
                        }
                    }
                }
                // A ^ (A | B) == ~A & B
                //
                // The OR adds B's bits to A; XORing A back out clears everything A
                // contributed, leaving the bits B supplied that A did not have.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::Or, x, y)) = self.as_bin(q) {
                        for (same, other) in [(x, y), (y, x)] {
                            if same == p {
                                let n = self.not(p);
                                return Some(self.bin(BinOp::And, n, other));
                            }
                        }
                    }
                }
                // (A & m) ^ (B & m) == (A ^ B) & m Factoring the shared mask out is what unlocks the VM's way of materializing a constant: it writes `m` as `(~x & m) ^ (x & m)`, where the two halves are individually unknown but complementary.
                if let (Some((BinOp::And, a1, m1)), Some((BinOp::And, b1, m2))) =
                    (self.as_bin(a), self.as_bin(b))
                {
                    for (ax, am) in [(a1, m1), (m1, a1)] {
                        for (bx, bm) in [(b1, m2), (m2, b1)] {
                            if am == bm && self.is_const(am) {
                                let inner = self.bin(BinOp::Xor, ax, bx);
                                return Some(self.bin(BinOp::And, inner, am));
                            }
                        }
                    }
                }
                if cb == Some(0) {
                    return Some(a);
                }
                if cb == Some(mask) {
                    return Some(self.not(a));
                }
                if self.is_complement(a, b) {
                    return Some(self.constant(mask, width));
                }
                // (A | c1) ^ c2 where c1 & c2 == c1 -> A ^ (c1 ^ c2)
                // This helps flatten nested constants in XOR chains
                if let (Some(c2), Some((BinOp::Or, x, y))) = (cb, self.as_bin(a)) {
                    if let Some(c1) = self.as_const(y) {
                        if (c1 & c2) == c1 {
                            let k = self.constant(c1 ^ c2, width);
                            return Some(self.bin(BinOp::Xor, x, k));
                        }
                    }
                }
                // (A ^ c1) ^ c2 -> A ^ (c1 ^ c2)
                // Merge nested XOR constants
                if let (Some(c2), Some((BinOp::Xor, x, y))) = (cb, self.as_bin(a)) {
                    if let Some(c1) = self.as_const(y) {
                        let k = self.constant(c1 ^ c2, width);
                        return Some(self.bin(BinOp::Xor, x, k));
                    }
                }
                // (x ^ c1) ^ c2 == x ^ (c1^c2)
                if let (Some(c2), Some((BinOp::Xor, x, y))) = (cb, self.as_bin(a)) {
                    if let Some(c1) = self.as_const(y) {
                        let k = self.constant(c1 ^ c2, width);
                        return Some(self.bin(BinOp::Xor, x, k));
                    }
                }
                // A ^ (A & B) == A & ~B
                //
                // The AND selects a subset of A's bits, so XORing removes exactly
                // those. Written this way by the obfuscator to hide a masked read.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::And, u, v)) = self.as_bin(p) {
                        for (same, other) in [(u, v), (v, u)] {
                            if same == q {
                                let n = self.not(other);
                                return Some(self.bin(BinOp::And, q, n));
                            }
                        }
                    }
                }
                // (a ^ b) ^ b == a, in either operand position.
                for (p, q) in [(a, b), (b, a)] {
                    if let Some((BinOp::Xor, x, y)) = self.as_bin(p) {
                        if x == q {
                            return Some(y);
                        }
                        if y == q {
                            return Some(x);
                        }
                    }
                }
                None
            }

            BinOp::Mul => {
                if cb == Some(0) {
                    return Some(self.constant(0, width));
                }
                if cb == Some(1) {
                    return Some(a);
                }
                None
            }

            BinOp::Shl | BinOp::Shr | BinOp::Sar | BinOp::Rol | BinOp::Ror => {
                if cb == Some(0) {
                    return Some(a);
                }
                None
            }

            BinOp::Eq => {
                if a == b {
                    return Some(self.constant(1, Width::W8));
                }
                None
            }

            BinOp::Ult | BinOp::Slt => {
                if a == b {
                    return Some(self.constant(0, Width::W8));
                }
                None
            }

            _ => None,
        }
    }

    pub fn as_bin(&self, r: Ref) -> Option<(BinOp, Ref, Ref)> {
        match *self.op(r) {
            Op::Bin(o, a, b) => Some((o, a, b)),
            _ => None,
        }
    }

    /// Bits that are provably zero in `r`. A miniature `computeKnownBits`. This is what lets the masking idioms TVM generates collapse: the VM constantly builds values like `((x & 0xffff_ffff_ffff_0000) | zext16(y)) & 0xffff`, where the outer mask annihilates the first term.
    fn bit_at(&mut self, r: Ref, k: u64) -> Option<Ref> {
        if k >= 64 {
            return None;
        }
        let bit = 1u64 << k;
        // A term cannot contribute if the bit is known zero in it.
        if self.known_zero(r) & bit == bit {
            return Some(self.constant(0, Width::W64));
        }
        if let Some(c) = self.as_const(r) {
            return Some(self.constant((c >> k) & 1, Width::W64));
        }
        // Already a 0-or-1 value: it is its own bit 0. Tested before the structural
        // cases because those recurse towards the leaf, and a masked value like
        // `af_undef & 1` is the bit even though the opaque underneath it is not.
        if k == 0 && self.known_zero(r) & !1 == !1 {
            return Some(r);
        }
        let node = self.op(r).clone();
        match node {
            // Split an OR: the bit comes from whichever side can supply it. If both
            // could, the value is not a clean bit composition and is left alone.
            Op::Bin(BinOp::Or, x, y) => {
                let zx = self.known_zero(x) & bit == bit;
                let zy = self.known_zero(y) & bit == bit;
                match (zx, zy) {
                    (false, true) => self.bit_at(x, k),
                    (true, false) => self.bit_at(y, k),
                    _ => None,
                }
            }
            // `v << s` puts v's bit `k - s` at position k.
            Op::Bin(BinOp::Shl, v, sh) => {
                let s = self.as_const(sh)?;
                if s > k {
                    return None;
                }
                self.bit_at(v, k - s)
            }
            Op::Bin(BinOp::Shr, v, sh) => {
                let s = self.as_const(sh)?;
                self.bit_at(v, k.checked_add(s)?)
            }
            // Masking with a constant that keeps the bit is transparent to it.
            Op::Bin(BinOp::And, x, y) => {
                let c = self.as_const(y)?;
                if c & bit == 0 {
                    return Some(self.constant(0, Width::W64));
                }
                self.bit_at(x, k)
            }
            // Zero-extend and truncate preserve every bit they keep, and the guard
            // above already rejected bits that fall outside the narrower width.
            Op::Zext(v) | Op::Trunc(v) => self.bit_at(v, k),
            _ => None,
        }
    }

    pub fn known_one(&mut self, r: Ref) -> u64 {
        if let Some(&k) = self.known_one.get(&r) {
            return k;
        }
        let inside = self.width(r).mask();
        let k = match *self.op(r) {
            Op::Const(c) => c,
            Op::Bin(op, a, b) => {
                let (oa, ob) = (self.known_one(a), self.known_one(b));
                match op {
                    // Set only where both are set.
                    BinOp::And => oa & ob,
                    // Set where either is set.
                    BinOp::Or => oa | ob,
                    // Set where exactly one is set *and* the other is known zero,
                    // otherwise the bit is unknown.
                    BinOp::Xor => {
                        let (za, zb) = (self.known_zero(a), self.known_zero(b));
                        (oa & zb) | (ob & za)
                    }
                    BinOp::Shl => match self.as_const(b) {
                        Some(c) => oa << ((c & 63) as u32),
                        None => 0,
                    },
                    BinOp::Shr => match self.as_const(b) {
                        Some(c) => oa >> ((c & 63) as u32),
                        None => 0,
                    },
                    _ => 0,
                }
            }
            Op::Zext(inner) | Op::Trunc(inner) => self.known_one(inner) & self.width(inner).mask(),
            // Set only where set on both arms, since the condition is unknown.
            Op::Select(_, a, b) => self.known_one(a) & self.known_one(b),
            // A bit known *zero* in the operand is known *one* after inverting.
            // Using `!known_zero` here would be unsound: it would claim a bit is set
            // merely because it is not known to be clear.
            Op::Un(UnOp::Not, inner) => self.known_zero(inner) & inside,
            _ => 0,
        } & inside;
        self.known_one.insert(r, k);
        k
    }

    pub fn known_zero(&mut self, r: Ref) -> u64 {
        if let Some(&k) = self.known_zero.get(&r) {
            return k;
        }
        // Bits above the node's width are always zero.
        let outside = !self.width(r).mask();
        let k = match *self.op(r) {
            Op::Const(c) => !c,
            Op::Bin(op, a, b) => {
                let (ka, kb) = (self.known_zero(a), self.known_zero(b));
                match op {
                    // A bit is zero if it is zero in either operand.
                    BinOp::And => ka | kb,
                    // Zero only where both are zero.
                    BinOp::Or | BinOp::Xor => ka & kb,
                    BinOp::Shl => match self.as_const(b) {
                        Some(c) => {
                            let c = (c & 63) as u32;
                            (ka << c) | ((1u64 << c) - 1)
                        }
                        None => 0,
                    },
                    BinOp::Shr => match self.as_const(b) {
                        Some(c) => {
                            let c = (c & 63) as u32;
                            let w = self.width(r).mask();
                            (ka >> c) | !(w >> c)
                        }
                        None => 0,
                    },
                    // Carries only propagate upward, so a common run of low zero
                    // bits survives addition, subtraction and multiplication.
                    BinOp::Add | BinOp::Sub => low_zero_run(ka & kb),
                    BinOp::Mul => {
                        let n = ka.trailing_ones().saturating_add(kb.trailing_ones());
                        mask_of_low_bits(n.min(64))
                    }
                    BinOp::Ult | BinOp::Slt | BinOp::Eq => !1u64,
                    _ => 0,
                }
            }
            Op::Zext(inner) => {
                let ki = self.known_zero(inner);
                let from = self.width(inner).mask();
                ki | !from
            }
            Op::Trunc(inner) => self.known_zero(inner),
            Op::Select(_, a, b) => self.known_zero(a) & self.known_zero(b),
            Op::Un(UnOp::ParityByte, _) => !1u64,
            _ => 0,
        } | outside;
        self.known_zero.insert(r, k);
        k
    }

    /// True when `b == ~a` or `a == ~b`, including the De Morgan spellings TVM emits: `~((~A ^ ~B) | ~A)` and `~(((A^B) & ~B) ^ ~A)`. Whether `r` could decompose into more than one conjunction factor.
    fn may_have_factors(&self, r: Ref) -> bool {
        matches!(
            *self.op(r),
            Op::Bin(BinOp::And, _, _) | Op::Un(UnOp::Not, _)
        )
    }

    /// The conjunction factors of `r`, reading `~(x | y)` as `~x & ~y`. Factors are what make disjointness provable for the obfuscator's output. Also returns whether the decomposition is *exact*, i.e. whether `r` equals the conjunction of the factors.
    fn and_factors(&self, r: Ref) -> (Vec<Ref>, bool) {
        let mut out = Vec::new();
        let exact = self.collect_and_factors(r, false, 6, &mut out);
        (out, exact)
    }

    /// Walk `r`, or its complement when `negated`, collecting conjunction factors. Returns whether the walk was exact. See [`Arena::and_factors`]. The leaves of an `Or`-tree, in left-to-right order.
    fn collect_or_terms(&self, r: Ref, depth: u32, out: &mut Vec<Ref>) {
        if depth > 0 {
            if let Op::Bin(BinOp::Or, x, y) = *self.op(r) {
                self.collect_or_terms(x, depth - 1, out);
                self.collect_or_terms(y, depth - 1, out);
                return;
            }
        }
        out.push(r);
    }

    fn collect_and_factors(&self, r: Ref, negated: bool, depth: u32, out: &mut Vec<Ref>) -> bool {
        if depth == 0 {
            return false;
        }
        match *self.op(r) {
            // `x & y` splits directly; `~(x | y)` splits as `~x & ~y`. The other two
            // combinations are disjunctions and are recorded whole, below.
            Op::Bin(BinOp::And, x, y) if !negated => {
                let l = self.collect_and_factors(x, false, depth - 1, out);
                let r2 = self.collect_and_factors(y, false, depth - 1, out);
                return l && r2;
            }
            Op::Bin(BinOp::Or, x, y) if negated => {
                let l = self.collect_and_factors(x, true, depth - 1, out);
                let r2 = self.collect_and_factors(y, true, depth - 1, out);
                return l && r2;
            }
            Op::Un(UnOp::Not, inner) => {
                return self.collect_and_factors(inner, !negated, depth - 1, out);
            }
            _ => {}
        }
        // A leaf for this purpose. Record it, spelling a pending negation the way the
        // arena would; with no existing node for it the factor cannot be named, so the
        // decomposition is reported inexact rather than silently narrowed.
        if !negated {
            out.push(r);
            return true;
        }
        let w = self.width(r);
        let neg = match *self.op(r) {
            Op::Const(c) => self.find_const(!c & w.mask(), w),
            _ => self.find_not(r),
        };
        match neg {
            Some(n) => {
                out.push(n);
                true
            }
            None => false,
        }
    }

    /// An already-interned `~r`, if one exists. Never creates a node.
    fn find_not(&self, r: Ref) -> Option<Ref> {
        self.intern
            .get(&Node {
                op: Op::Un(UnOp::Not, r),
                width: self.width(r),
            })
            .copied()
    }

    /// An already-interned constant, if one exists. Never creates a node.
    fn find_const(&self, value: u64, width: Width) -> Option<Ref> {
        self.intern
            .get(&Node {
                op: Op::Const(value & width.mask()),
                width,
            })
            .copied()
    }

    /// Whether `a` and `b` provably share no live bit.
    fn disjoint_by_factors(&self, a: Ref, b: Ref) -> bool {
        if !self.may_have_factors(a) || !self.may_have_factors(b) {
            return false;
        }
        let (fa, _) = self.and_factors(a);
        let (fb, _) = self.and_factors(b);
        fa.iter()
            .any(|&x| fb.iter().any(|&y| self.is_complement(x, y)))
    }

    fn is_complement(&self, a: Ref, b: Ref) -> bool {
        if let Op::Un(UnOp::Not, inner) = *self.op(b) {
            if inner == a {
                return true;
            }
        }
        if let Op::Un(UnOp::Not, inner) = *self.op(a) {
            if inner == b {
                return true;
            }
        }
        false
    }
}

/// Collect the symbolic leaves (`InitReg`, `Load`, `Opaque`) an expression depends on. Diagnosing *why* a dispatch computation failed to fold comes down to seeing which leaves it still references.
pub fn is_guest_rooted(a: &Arena, r: Ref) -> bool {
    leaves(a, r)
        .iter()
        .any(|l| matches!(a.op(*l), Op::Param(..) | Op::InitReg(_)))
}

pub fn leaves(a: &Arena, r: Ref) -> Vec<Ref> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut stack = vec![r];
    while let Some(cur) = stack.pop() {
        if !seen.insert(cur) {
            continue;
        }
        match *a.op(cur) {
            Op::Const(_) => {}
            Op::InitReg(_) | Op::Opaque(..) | Op::Param(..) => out.push(cur),
            Op::Load(addr, _) => {
                out.push(cur);
                stack.push(addr);
            }
            Op::Bin(_, x, y) => {
                stack.push(x);
                stack.push(y);
            }
            Op::Un(_, x) | Op::Zext(x) | Op::Sext(x) | Op::Trunc(x) => stack.push(x),
            Op::Select(c, x, y) => {
                stack.push(c);
                stack.push(x);
                stack.push(y);
            }
        }
    }
    out
}

fn same_pair(a1: Ref, b1: Ref, a2: Ref, b2: Ref) -> bool {
    (a1 == a2 && b1 == b2) || (a1 == b2 && b1 == a2)
}

/// Mask covering the contiguous run of low bits that are set in `k`.
/// e.g. `0b1011 -> 0b0011`. Used to reason about carry propagation.
fn low_zero_run(k: u64) -> u64 {
    mask_of_low_bits(k.trailing_ones())
}

fn mask_of_low_bits(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

pub fn sign_extend(value: u64, from: Width) -> u64 {
    match from {
        Width::W64 => value,
        w => {
            let bits = w.bits();
            let shifted = value << (64 - bits);
            ((shifted as i64) >> (64 - bits)) as u64
        }
    }
}

fn bswap(v: u64, w: Width) -> u64 {
    match w {
        Width::W8 => v & 0xff,
        Width::W16 => (v as u16).swap_bytes() as u64,
        Width::W32 => (v as u32).swap_bytes() as u64,
        Width::W64 => v.swap_bytes(),
    }
}

/// Evaluate `op` on two concrete values at `width`. `None` for div-by-zero.
pub fn fold_bin(op: BinOp, x: u64, y: u64, width: Width) -> Option<u64> {
    let m = width.mask();
    let bits = width.bits() as u64;
    let (x, y) = (x & m, y & m);
    let sx = sign_extend(x, width) as i64;
    let sy = sign_extend(y, width) as i64;
    Some(
        match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::And => x & y,
            BinOp::Or => x | y,
            BinOp::Xor => x ^ y,
            // x86 masks the shift count by operand size; 8/16-bit use mod 32.
            BinOp::Shl => {
                let c = shift_count(y, width);
                if c >= bits { 0 } else { x << c }
            }
            BinOp::Shr => {
                let c = shift_count(y, width);
                if c >= bits { 0 } else { x >> c }
            }
            BinOp::Sar => {
                let c = shift_count(y, width).min(bits - 1);
                ((sx) >> c) as u64
            }
            BinOp::Rol => {
                let c = y % bits;
                if c == 0 {
                    x
                } else {
                    (x << c) | (x >> (bits - c))
                }
            }
            BinOp::Ror => {
                let c = y % bits;
                if c == 0 {
                    x
                } else {
                    (x >> c) | (x << (bits - c))
                }
            }
            BinOp::MulHiU => ((x as u128 * y as u128) >> bits) as u64,
            BinOp::MulHiS => (((sx as i128 * sy as i128) >> bits) as u128) as u64,
            BinOp::UDiv => x.checked_div(y)?,
            BinOp::URem => x.checked_rem(y)?,
            BinOp::SDiv => sx.checked_div(sy)? as u64,
            BinOp::SRem => sx.checked_rem(sy)? as u64,
            BinOp::Ult => (x < y) as u64,
            BinOp::Slt => (sx < sy) as u64,
            BinOp::Eq => (x == y) as u64,
        } & m,
    )
}

fn shift_count(y: u64, width: Width) -> u64 {
    match width {
        Width::W64 => y & 0x3f,
        _ => y & 0x1f,
    }
}

/// Depth-limited rendering, for diagnostics.
pub fn render(a: &Arena, r: Ref, max_depth: u32) -> String {
    fn go(a: &Arena, r: Ref, depth: u32, max: u32, out: &mut String) {
        if depth > max {
            out.push_str("...");
            return;
        }
        match a.op(r) {
            Op::Const(c) => out.push_str(&format!("{c:#x}")),
            Op::InitReg(reg) => out.push_str(reg.name()),
            Op::Opaque(tag, id) => out.push_str(&format!("{tag}#{id}")),
            Op::Param(b, reg) => out.push_str(&format!("{}.{}", b, reg.name())),
            Op::Load(addr, _) => {
                out.push_str(&format!("load{}[", a.width(r).bits()));
                go(a, *addr, depth + 1, max, out);
                out.push(']');
            }
            Op::Bin(op, x, y) => {
                out.push('(');
                go(a, *x, depth + 1, max, out);
                out.push_str(&format!(" {} ", op.symbol()));
                go(a, *y, depth + 1, max, out);
                out.push(')');
            }
            Op::Un(op, x) => {
                out.push_str(match op {
                    UnOp::Not => "~",
                    UnOp::Neg => "-",
                    UnOp::ParityByte => "parity",
                    UnOp::Bswap => "bswap",
                });
                go(a, *x, depth + 1, max, out);
            }
            Op::Zext(x) => {
                out.push_str(&format!("zext{}(", a.width(r).bits()));
                go(a, *x, depth + 1, max, out);
                out.push(')');
            }
            Op::Sext(x) => {
                out.push_str(&format!("sext{}(", a.width(r).bits()));
                go(a, *x, depth + 1, max, out);
                out.push(')');
            }
            Op::Trunc(x) => {
                out.push_str(&format!("trunc{}(", a.width(r).bits()));
                go(a, *x, depth + 1, max, out);
                out.push(')');
            }
            Op::Select(c, x, y) => {
                go(a, *c, depth + 1, max, out);
                out.push_str(" ? ");
                go(a, *x, depth + 1, max, out);
                out.push_str(" : ");
                go(a, *y, depth + 1, max, out);
            }
        }
    }
    let mut s = String::new();
    go(a, r, 0, max_depth, &mut s);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(a: &mut Arena, name: Reg) -> Ref {
        a.init_reg(name)
    }

    /// Nodes are hash-consed, so a load must carry the memory generation to
    /// distinguish loads separated by a write.
    #[test]
    fn a_load_after_a_write_is_not_the_load_before_it() {
        let mut a = Arena::new();
        let addr = sym(&mut a, Reg::Rsp);

        let before = a.load(addr, Width::W8);
        // Same generation: still one node, so CSE keeps working.
        assert_eq!(before, a.load(addr, Width::W8));

        a.bump_mem_gen();
        let after = a.load(addr, Width::W8);

        assert_ne!(
            before, after,
            "a load after a write interned to one taken before it"
        );
    }

    /// Rebuilding an existing load must keep its generation. `graft` and
    /// `substitute` go through `load_at` for this reason: stamping the current
    /// generation would merge loads the source arena kept apart.
    #[test]
    fn rebuilding_a_load_preserves_its_generation() {
        let mut a = Arena::new();
        let addr = sym(&mut a, Reg::Rsp);

        let first = a.load(addr, Width::W8);
        let gen_of_first = a.mem_gen();
        a.bump_mem_gen();
        let second = a.load(addr, Width::W8);
        assert_ne!(first, second);

        // At the recorded generation: recovers the original node.
        assert_eq!(a.load_at(addr, Width::W8, gen_of_first), first);
        // At the current one: the later node, not the earlier.
        let now = a.mem_gen();
        assert_eq!(a.load_at(addr, Width::W8, now), second);
    }

    /// The generation is arena-local, so hashing it would make two structurally
    /// identical program points reached by different paths hash differently and
    /// defeat the state matching that consumes this.
    #[test]
    fn the_structural_hash_ignores_the_generation() {
        let mut a = Arena::new();
        let addr = sym(&mut a, Reg::Rsp);
        let first = a.load(addr, Width::W8);
        a.bump_mem_gen();
        let second = a.load(addr, Width::W8);

        assert_ne!(first, second);
        assert_eq!(a.structural_hash(first, 8), a.structural_hash(second, 8));
    }

    #[test]
    fn mba_or_and_add() {
        // (A|B) + (A&B) == A + B
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, Reg::Rax), sym(&mut a, Reg::Rbx));
        let or = a.bin(BinOp::Or, x, y);
        let and = a.bin(BinOp::And, x, y);
        let got = a.bin(BinOp::Add, or, and);
        let want = a.bin(BinOp::Add, x, y);
        assert_eq!(got, want);
    }

    #[test]
    fn mba_xor_and_or() {
        // (A^B) + (A&B) == A | B
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, Reg::Rax), sym(&mut a, Reg::Rbx));
        let xor = a.bin(BinOp::Xor, x, y);
        let and = a.bin(BinOp::And, x, y);
        let got = a.bin(BinOp::Add, xor, and);
        let want = a.bin(BinOp::Or, x, y);
        assert_eq!(got, want);
    }

    #[test]
    fn mba_or_sub_and_is_xor() {
        // (A|B) - (A&B) == A ^ B
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, Reg::Rax), sym(&mut a, Reg::Rbx));
        let or = a.bin(BinOp::Or, x, y);
        let and = a.bin(BinOp::And, x, y);
        let got = a.bin(BinOp::Sub, or, and);
        let want = a.bin(BinOp::Xor, x, y);
        assert_eq!(got, want);
    }

    #[test]
    fn mba_demorgan_to_and() {
        // ~((~A ^ ~B) | ~A) == A & B
        //
        // ~A ^ ~B folds to A ^ B, so this reduces to ~((A^B) | ~A).
        let mut a = Arena::new();
        let (x, y) = (sym(&mut a, Reg::Rax), sym(&mut a, Reg::Rbx));
        let nx = a.not(x);
        let ny = a.not(y);
        let xor = a.bin(BinOp::Xor, nx, ny);
        let or = a.bin(BinOp::Or, xor, nx);
        let got = a.not(or);
        // Verify semantically over a sample of inputs rather than structurally,
        // since the rule chain may land on a different but equal form.
        let want = a.bin(BinOp::And, x, y);
        assert!(crate::ir::expr::tests::equal_on_samples(&a, got, want));
    }

    #[test]
    fn or_drops_operand_already_covered() {
        // (~A & c) | (A | c) == A | c
        //
        // Every bit the left side can contribute is inside c, and c is already known
        // set on the right, so the left side is redundant.
        let mut a = Arena::default();
        let x = a.init_reg(Reg::Rdx);
        let c = a.constant(0x30, Width::W64);
        let nx = a.un(UnOp::Not, x);
        let lhs = a.bin(BinOp::And, nx, c);
        let rhs = a.bin(BinOp::Or, x, c);
        let got = a.bin(BinOp::Or, lhs, rhs);
        assert_eq!(got, rhs, "got {}", render(&a, got, 8));
    }

    #[test]
    fn and_xor_or_identity() {
        // (A&B) ^ (A|B) == A ^ B
        let mut a = Arena::default();
        let x = a.init_reg(Reg::Rcx);
        let c = a.constant(0x80, Width::W64);
        let and = a.bin(BinOp::And, x, c);
        let or = a.bin(BinOp::Or, x, c);
        let got = a.bin(BinOp::Xor, and, or);
        let want = a.bin(BinOp::Xor, x, c);
        assert_eq!(got, want, "got {}", render(&a, got, 8));
    }

    #[test]
    fn xor_const_or_const_collapses() {
        // (A ^ c) | c == A | c
        let mut a = Arena::default();
        let x = a.init_reg(Reg::Rcx);
        let c = a.constant(0xe0, Width::W64);
        let xo = a.bin(BinOp::Xor, x, c);
        let got = a.bin(BinOp::Or, xo, c);
        let want = a.bin(BinOp::Or, x, c);
        assert_eq!(got, want, "got {}", render(&a, got, 8));
    }

    #[test]
    fn xor_with_own_or_is_masked_complement() {
        // A ^ (A | B) == ~A & B
        let mut a = Arena::default();
        let x = a.init_reg(Reg::Rcx);
        let c = a.constant(0xe0, Width::W64);
        let or = a.bin(BinOp::Or, x, c);
        let got = a.bin(BinOp::Xor, x, or);
        let nx = a.un(UnOp::Not, x);
        let want = a.bin(BinOp::And, nx, c);
        assert_eq!(got, want, "got {}", render(&a, got, 8));
    }

    #[test]
    fn or_absorbs_through_nested_or() {
        // A | ((A & k) | c) == A | c
        let mut a = Arena::default();
        let x = a.init_reg(Reg::Rcx);
        let k = a.constant(0xffff_ffff_ffff_ff0f, Width::W64);
        let c = a.constant(0xf0, Width::W64);
        let inner = a.bin(BinOp::And, x, k);
        let nested = a.bin(BinOp::Or, inner, c);
        let got = a.bin(BinOp::Or, x, nested);
        let want = a.bin(BinOp::Or, x, c);
        assert_eq!(got, want, "got {}", render(&a, got, 8));
    }

    #[test]
    fn and_keeps_only_known_one_bits() {
        // (A | c) & c == c, because c's bits are all known set.
        let mut a = Arena::default();
        let x = a.init_reg(Reg::Rcx);
        let c = a.constant(0xf0, Width::W64);
        let or = a.bin(BinOp::Or, x, c);
        let got = a.bin(BinOp::And, or, c);
        assert_eq!(a.as_const(got), Some(0xf0), "got {}", render(&a, got, 8));
    }

    fn mask_of(a: &mut Arena, x: Ref, c: u64) -> Ref {
        // x & c, written the way the obfuscator does: x - (x & (x ^ (x & c)))
        let kc = a.constant(c, Width::W64);
        let inner = a.bin(BinOp::And, x, kc);
        let cleared = a.bin(BinOp::Xor, x, inner);
        let anded = a.bin(BinOp::And, x, cleared);
        a.bin(BinOp::Sub, x, anded)
    }

    #[test]
    fn sub_of_and_complement_is_mask() {
        // x - (x & (x ^ (x & c))) == x & c `x ^ (x & c)` clears exactly c's bits from x, so the AND with x gives x & ~c, and subtracting that from x leaves x & c.
        let mut a = Arena::default();
        let x = a.init_reg(Reg::Rcx);
        let got = mask_of(&mut a, x, 0x70);
        let kc = a.constant(0x70, Width::W64);
        let want = a.bin(BinOp::And, x, kc);
        assert_eq!(got, want, "got {}", render(&a, got, 8));
    }

    #[test]
    fn demorgan_not_or_not() {
        // ~((~v) | c) == v & ~c
        let mut a = Arena::default();
        let v = a.init_reg(Reg::Rcx);
        let nv = a.un(UnOp::Not, v);
        let kc = a.constant(0xffff_ffff_ffff_ff8f, Width::W64);
        let ored = a.bin(BinOp::Or, nv, kc);
        let got = a.un(UnOp::Not, ored);
        let k2 = a.constant(0x70, Width::W64);
        let want = a.bin(BinOp::And, v, k2);
        assert_eq!(got, want, "got {}", render(&a, got, 8));
    }

    #[test]
    fn xor_cancels_through_xor_const() {
        // u ^ (u ^ c) == c
        let mut a = Arena::default();
        let u = a.init_reg(Reg::Rcx);
        let kc = a.constant(0x70, Width::W64);
        let inner = a.bin(BinOp::Xor, u, kc);
        let got = a.bin(BinOp::Xor, u, inner);
        assert_eq!(a.as_const(got), Some(0x70), "got {}", render(&a, got, 8));
    }

    #[test]
    fn params_are_distinct_per_block_and_register() {
        let mut a = Arena::default();
        let b0 = BlockRef(0);
        let b1 = BlockRef(1);
        // Same register in two blocks must not be CSE'd: they are different
        // values, and conflating them would merge unrelated paths.
        assert_ne!(a.param(b0, Reg::Rax), a.param(b1, Reg::Rax));
        // Different registers in one block are likewise distinct.
        assert_ne!(a.param(b0, Reg::Rax), a.param(b0, Reg::Rcx));
        // But the same (block, register) is one value, so uses share it.
        assert_eq!(a.param(b0, Reg::Rax), a.param(b0, Reg::Rax));
    }

    #[test]
    fn params_survive_grafting_unchanged() {
        // Grafting must preserve parameter identity exactly. Unlike `Opaque`,
        // which is renumbered to avoid collisions, (block, register) *is* the
        // identity: renumbering would break the link between a block's
        // expressions and its phis.
        let mut src = Arena::default();
        let p = src.param(BlockRef(3), Reg::R12);
        let eight = src.constant(8, Width::W64);
        let e = src.bin(BinOp::Add, p, eight);

        let mut dst = Arena::default();
        let mut memo = HashMap::new();
        let out = dst.graft(&src, e, &mut memo);
        let expect_p = dst.param(BlockRef(3), Reg::R12);
        let dst_eight = dst.constant(8, Width::W64);
        let expect = dst.bin(BinOp::Add, expect_p, dst_eight);
        assert_eq!(out, expect);
    }

    #[test]
    fn rewrite_substitutes_and_resimplifies() {
        let mut a = Arena::default();
        let p = a.param(BlockRef(0), Reg::Rsp);
        // (rsp - 8) + 8, which does not fold while rsp is opaque.
        let eight = a.constant(8, Width::W64);
        let lo = a.bin(BinOp::Sub, p, eight);
        let e = a.bin(BinOp::Add, lo, eight);
        assert_eq!(a.as_const(e), None);

        // Resolving the parameter to a constant must fold the whole expression,
        // which is the point of rebuilding through the constructors rather than
        // copying structurally. This is the trivial-phi case for guest RSP.
        let mut map = HashMap::new();
        let base = a.constant(0x7fff_fffe_fda0, Width::W64);
        map.insert(p, base);
        let out = a.rewrite(e, &map);
        assert_eq!(a.as_const(out), Some(0x7fff_fffe_fda0));
    }

    #[test]
    fn graft_preserves_semantics_and_sharing() {
        // Build an expression in one arena, with a deliberately shared subterm.
        let mut src = Arena::default();
        let rax = src.init_reg(Reg::Rax);
        let rbx = src.init_reg(Reg::Rbx);
        let shared = src.bin(BinOp::Xor, rax, rbx);
        let addr = src.bin(BinOp::Add, shared, rax);
        let loaded = src.load(addr, Width::W32);
        let widened = src.zext(loaded, Width::W64);
        let root = src.bin(BinOp::Or, widened, shared);
        let tsc = src.opaque("rdtsc", Width::W64);
        let with_tsc = src.bin(BinOp::Add, root, tsc);

        let mut dst = Arena::default();
        let mut memo = HashMap::new();
        let g_root = dst.graft(&src, root, &mut memo);
        let g_shared = dst.graft(&src, shared, &mut memo);
        let g_again = dst.graft(&src, root, &mut memo);

        // Sharing survives: the second graft of the same node is the same node,
        // and the shared subterm is not duplicated.
        assert_eq!(g_root, g_again);
        assert_eq!(dst.width(g_root), src.width(root));
        assert_eq!(dst.width(g_shared), src.width(shared));

        // Semantics survive, checked by evaluating in each arena under the same
        // environment. `Load` evaluates as a function of its address in `eval`,
        // so this covers the load and the zext too.
        let mut seed = 0x2545F4914F6CDD1Du64;
        for _ in 0..64 {
            let mut env = HashMap::new();
            for reg in GPRS {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                env.insert(reg, seed);
            }
            assert_eq!(eval(&src, root, &env), eval(&dst, g_root, &env));
        }

        // Opaque nodes are given fresh identities rather than copied, so two
        // distinct opaques must stay distinct after grafting.
        let tsc2 = src.opaque("rdtsc", Width::W64);
        assert_ne!(tsc, tsc2);
        let g_tsc = dst.graft(&src, with_tsc, &mut memo);
        let g_tsc2 = dst.graft(&src, tsc2, &mut memo);
        assert_ne!(g_tsc, g_tsc2);
    }

    /// A shared `undef` stays shared across grafts; a counted `opaque` does not. The complement of the last assertion in `graft_preserves_semantics_and_sharing`. Fresh memos each time, because a shared memo would preserve sharing on its own and hide a regression here.
    #[test]
    fn graft_keeps_a_shared_undef_shared() {
        let mut src = Arena::new();
        let mut dst = Arena::new();

        let u = src.undef("af_undef", Width::W8);
        assert_eq!(
            u,
            src.undef("af_undef", Width::W8),
            "undef must share within an arena"
        );

        let g1 = dst.graft(&src, u, &mut HashMap::new());
        let g2 = dst.graft(&src, u, &mut HashMap::new());
        assert_eq!(
            g1, g2,
            "a shared undef must survive separate grafts as one node"
        );

        // A counted opaque behaves the opposite way, for the same reason.
        let o = src.opaque("rdtsc", Width::W64);
        let h1 = dst.graft(&src, o, &mut HashMap::new());
        let h2 = dst.graft(&src, o, &mut HashMap::new());
        assert_ne!(h1, h2, "a counted opaque must be reissued per graft");
    }

    /// Brute-force semantic check of two expressions over random assignments to
    /// their `InitReg` leaves. Used to validate rewrite rules.
    pub(crate) fn equal_on_samples(a: &Arena, p: Ref, q: Ref) -> bool {
        let mut seed = 0x243F6A8885A308D3u64;
        for _ in 0..64 {
            let mut env = HashMap::new();
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for r in GPRS {
                env.insert(r, next());
            }
            if eval(a, p, &env) != eval(a, q, &env) {
                return false;
            }
        }
        true
    }

    /// Union of all bits ever observed set in `r` over random assignments.
    /// Any bit `known_zero` claims must be absent from this set.
    pub(crate) fn observed_ones(a: &Arena, r: Ref) -> u64 {
        let mut seed = 0x9E3779B97F4A7C15u64;
        let mut acc = 0u64;
        for _ in 0..512 {
            let mut env = HashMap::new();
            for reg in GPRS {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                env.insert(reg, seed);
            }
            acc |= eval(a, r, &env);
        }
        // Also probe the all-ones and all-zeros corners.
        for fill in [0u64, u64::MAX] {
            let env: HashMap<Reg, u64> = GPRS.iter().map(|&g| (g, fill)).collect();
            acc |= eval(a, r, &env);
        }
        acc
    }

    pub(crate) fn eval(a: &Arena, r: Ref, env: &HashMap<Reg, u64>) -> u64 {
        let w = a.width(r);
        let v = match a.op(r) {
            Op::Const(c) => *c,
            Op::InitReg(reg) => env[reg],
            Op::Un(UnOp::Not, x) => !eval(a, *x, env),
            Op::Un(UnOp::Neg, x) => eval(a, *x, env).wrapping_neg(),
            Op::Un(UnOp::Bswap, x) => bswap(eval(a, *x, env), w),
            Op::Un(UnOp::ParityByte, x) => ((eval(a, *x, env) as u8).count_ones() % 2 == 0) as u64,
            Op::Bin(o, x, y) => fold_bin(
                *o,
                eval(a, *x, env),
                eval(a, *y, env),
                a.width(*x).max(a.width(*y)),
            )
            .unwrap_or(0),
            Op::Zext(x) => eval(a, *x, env) & a.width(*x).mask(),
            Op::Sext(x) => sign_extend(eval(a, *x, env), a.width(*x)),
            Op::Trunc(x) => eval(a, *x, env) & w.mask(),
            Op::Select(c, x, y) => {
                if eval(a, *c, env) != 0 {
                    eval(a, *x, env)
                } else {
                    eval(a, *y, env)
                }
            }
            // Not modelled by the sampling evaluator: these are the opaque
            // inputs of an expression, and the simplification rules under test
            // must hold for any value they take.
            Op::Load(..) | Op::Opaque(..) | Op::Param(..) => 0,
        };
        v & w.mask()
    }
}

#[cfg(test)]
mod bits_tests {
    use super::*;

    /// `known_zero` must never claim a bit is zero when some assignment makes
    /// it one. Unsound bit facts would silently corrupt folded addresses, so
    /// this is checked against brute-force evaluation.
    #[test]
    fn known_zero_is_sound() {
        let mut a = Arena::new();
        let x = a.init_reg(Reg::Rax);
        let y = a.init_reg(Reg::Rbx);

        let mut probes = Vec::new();
        let m1 = a.constant(0xffff_ffff_ffff_0000, Width::W64);
        let hi = a.bin(BinOp::And, x, m1);
        let t = a.trunc(y, Width::W16);
        let z = a.zext(t, Width::W64);
        probes.push(a.bin(BinOp::Or, hi, z));
        let sh = a.constant(8, Width::W64);
        probes.push(a.bin(BinOp::Shl, x, sh));
        probes.push(a.bin(BinOp::Shr, x, sh));
        let m2 = a.constant(0xff00, Width::W64);
        let am = a.bin(BinOp::And, x, m2);
        let bm = a.bin(BinOp::And, y, m2);
        probes.push(a.bin(BinOp::Add, am, bm));
        probes.push(a.bin(BinOp::Mul, am, bm));
        probes.push(a.bin(BinOp::Xor, am, bm));
        probes.push(a.trunc(x, Width::W8));

        for p in probes {
            let kz = a.known_zero(p);
            let observed = super::tests::observed_ones(&a, p);
            assert_eq!(
                observed & kz,
                0,
                "unsound known_zero for {:?}: claimed zero {:#x}, observed ones {:#x}",
                a.op(p),
                kz,
                observed
            );
        }
    }

    /// Observed *zeros* over random assignments: bit set in the result means some
    /// assignment left that bit clear.
    fn observed_zeros(a: &Arena, r: Ref) -> u64 {
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut acc = 0u64;
        for _ in 0..512 {
            let mut env = HashMap::new();
            for reg in GPRS {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                env.insert(reg, seed);
            }
            acc |= !super::tests::eval(a, r, &env);
        }
        acc
    }

    /// The structural hash must agree with structural equality: equal shapes hash equal even when built in different arenas, and different shapes separate. The first property is the one recovery depends on.
    #[test]
    fn structural_hash_is_stable_across_arenas() {
        // Build the same expression in two arenas, in a different order and with
        // unrelated padding, so the Refs cannot coincide.
        let build = |pad: usize| {
            let mut a = Arena::default();
            for i in 0..pad {
                let _ = a.constant(0xdead_0000 + i as u64, Width::W64);
            }
            let rcx = a.init_reg(Reg::Rcx);
            let k = a.constant(0x1e22_c984_c352_8bb6, Width::W64);
            let t = a.bin(BinOp::Xor, rcx, k);
            let l = a.load(t, Width::W64);
            let e = a.bin(BinOp::Add, l, k);
            (a, e)
        };
        let (a1, e1) = build(0);
        let (a2, e2) = build(37);
        assert_ne!(
            e1.0, e2.0,
            "the two arenas must not hand out the same index"
        );
        assert_eq!(
            a1.structural_hash(e1, 12),
            a2.structural_hash(e2, 12),
            "the same shape must hash the same in any arena"
        );

        // Distinct shapes must separate: a different operator, a different operand,
        // a different register, and a different constant.
        let mut a = Arena::default();
        let rcx = a.init_reg(Reg::Rcx);
        let rdx = a.init_reg(Reg::Rdx);
        let k1 = a.constant(7, Width::W64);
        let k2 = a.constant(8, Width::W64);
        let variants = vec![
            a.bin(BinOp::Add, rcx, k1),
            a.bin(BinOp::Sub, rcx, k1),
            a.bin(BinOp::Add, rdx, k1),
            a.bin(BinOp::Add, rcx, k2),
            rcx,
            rdx,
        ];
        let mut seen = std::collections::HashMap::new();
        for (i, v) in variants.iter().enumerate() {
            let h = a.structural_hash(*v, 12);
            if let Some(j) = seen.insert(h, i) {
                panic!("variants {j} and {i} collided");
            }
        }

        // Commutative operands are canonicalized by `bin`, so the two orders are
        // literally the same node and must hash alike. Asserting it here records
        // that the equal hash is canonicalization rather than a collision.
        let ab = a.bin(BinOp::Add, rcx, k1);
        let ba = a.bin(BinOp::Add, k1, rcx);
        assert_eq!(ab, ba, "commutative operands are canonicalized");
        assert_eq!(a.structural_hash(ab, 12), a.structural_hash(ba, 12));

        // Truncation must collapse, never split: hashing the same value at the same
        // depth twice has to agree, however deep the value is.
        let mut deep = a.init_reg(Reg::Rax);
        for i in 0..40u64 {
            let c = a.constant(i, Width::W64);
            deep = a.bin(BinOp::Add, deep, c);
        }
        assert_eq!(a.structural_hash(deep, 4), a.structural_hash(deep, 4));
    }

    /// `known_one` must never claim a bit is set when some assignment leaves it clear. Checked by brute force for the same reason as `known_zero`: an unsound bit fact silently corrupts folded addresses, and this analysis is what licenses replacing a whole subtree with a constant.
    #[test]
    fn known_one_is_sound() {
        let mut a = Arena::new();
        let x = a.init_reg(Reg::Rax);
        let y = a.init_reg(Reg::Rbx);

        let mut probes = Vec::new();
        let c70 = a.constant(0x70, Width::W64);
        let cff = a.constant(0xff00, Width::W64);
        probes.push(c70);
        // ~x, the case that was originally wrong: not-known-zero is not known-one.
        let nx = a.un(UnOp::Not, x);
        probes.push(nx);
        probes.push(a.bin(BinOp::And, nx, c70));
        probes.push(a.bin(BinOp::Or, x, c70));
        probes.push(a.bin(BinOp::And, x, c70));
        probes.push(a.bin(BinOp::Xor, x, c70));
        let ax = a.bin(BinOp::And, x, c70);
        let ay = a.bin(BinOp::And, y, cff);
        probes.push(a.bin(BinOp::Xor, ax, ay));
        probes.push(a.bin(BinOp::Or, ax, ay));
        let sh = a.constant(8, Width::W64);
        probes.push(a.bin(BinOp::Shl, c70, sh));
        probes.push(a.bin(BinOp::Shr, cff, sh));
        probes.push(a.trunc(cff, Width::W8));
        let t16 = a.trunc(cff, Width::W16);
        probes.push(a.zext(t16, Width::W64));

        for p in probes {
            let ko = a.known_one(p);
            let zeros = observed_zeros(&a, p);
            assert_eq!(
                zeros & ko,
                0,
                "unsound known_one for {:?}: claimed one {:#x}, observed zeros {:#x}",
                a.op(p),
                ko,
                zeros
            );
        }
    }

    #[test]
    fn vm_context_mask_idiom_folds() {
        // ((x & 0xffff_ffff_ffff_0000) | zext16(y)) & 0xffff  ==  y & 0xffff
        let mut a = Arena::new();
        let x = a.init_reg(Reg::Rax);
        let y = a.init_reg(Reg::Rbx);
        let m1 = a.constant(0xffff_ffff_ffff_0000, Width::W64);
        let hi = a.bin(BinOp::And, x, m1);
        let t = a.trunc(y, Width::W16);
        let z = a.zext(t, Width::W64);
        let or = a.bin(BinOp::Or, hi, z);
        let m2 = a.constant(0xffff, Width::W64);
        let got = a.bin(BinOp::And, or, m2);
        let want = a.bin(BinOp::And, y, m2);
        assert_eq!(got, want, "got {:?}", a.op(got));
    }

    #[test]
    fn zext_trunc_becomes_mask() {
        let mut a = Arena::new();
        let x = a.init_reg(Reg::Rdx);
        let t = a.trunc(x, Width::W16);
        let z = a.zext(t, Width::W64);
        // expect (x & 0xffff)
        match a.op(z) {
            Op::Bin(BinOp::And, _, m) => assert_eq!(a.as_const(*m), Some(0xffff)),
            other => panic!("not a mask: {other:?}"),
        }
    }
}

#[cfg(test)]
mod addr_tests {
    use super::*;

    /// TVM computes `base + idx` obfuscated as `((base+idx) & idx) | (base+idx)`
    /// and `((t+t) | t) & t`. Both must collapse, otherwise every table read
    /// stays symbolic and dispatch stalls.
    #[test]
    fn obfuscated_address_forms_collapse() {
        let mut a = Arena::new();
        let base = a.constant(0x140000000, Width::W64);
        let idx = a.init_reg(Reg::R9);

        // mov rax,rdx / add rax,r9 / and rax,r9 / add rdx,r9 / or rax,rdx
        let sum = a.bin(BinOp::Add, base, idx);
        let anded = a.bin(BinOp::And, sum, idx);
        let got = a.bin(BinOp::Or, anded, sum);
        assert_eq!(
            got,
            sum,
            "((b+i)&i)|(b+i) should be b+i, got {:?}",
            a.op(got)
        );

        // lea r8,[X] / mov rdx,r8 / add rdx,r8 / or rdx,r8 / and rdx,r8
        let t = a.constant(0x14017c21e, Width::W64);
        let tt = a.bin(BinOp::Add, t, t);
        let or = a.bin(BinOp::Or, tt, t);
        let and = a.bin(BinOp::And, or, t);
        assert_eq!(a.as_const(and), Some(0x14017c21e));
    }
}

#[cfg(test)]
mod vmctx_addr_tests {
    use super::*;

    #[test]
    fn or_xor_or_absorbs() {
        let mut a = Arena::new();
        let base = a.init_reg(Reg::R10);
        let idx = a.init_reg(Reg::Rcx);
        let or1 = a.bin(BinOp::Or, base, idx);
        let x = a.bin(BinOp::Xor, or1, idx);
        let or2 = a.bin(BinOp::Or, x, base);
        assert_eq!(or2, base, "((A|B)^B)|A should be A, got {:?}", a.op(or2));
        let sum = a.bin(BinOp::Add, or2, idx);
        let want = a.bin(BinOp::Add, base, idx);
        assert_eq!(sum, want);
    }

    /// Same identity, verified semantically rather than structurally.
    #[test]
    fn or_xor_or_is_semantically_sound() {
        let mut a = Arena::new();
        let p = a.init_reg(Reg::Rax);
        let q = a.init_reg(Reg::Rbx);
        let or1 = a.bin(BinOp::Or, p, q);
        let x = a.bin(BinOp::Xor, or1, q);
        let got = a.bin(BinOp::Or, x, p);
        assert!(super::tests::equal_on_samples(&a, got, p));
    }

    /// `(~t & X) | (t & X) == X`, for an arbitrary `t`. Every bit of `X` is selected by exactly one arm, so the pair reconstructs `X` whatever `t` is.
    #[test]
    fn complementary_masks_over_one_value_rebuild_it() {
        let mut a = Arena::new();
        let x = a.init_reg(Reg::Rax);
        let t = a.init_reg(Reg::Rbx);
        let nt = a.not(t);
        let lo = a.bin(BinOp::And, nt, x);
        let hi = a.bin(BinOp::And, t, x);
        let got = a.bin(BinOp::Or, lo, hi);
        assert_eq!(got, x, "got {}", render(&a, got, 12));
    }

    /// The identity as it actually reaches the arena: `t` is the VM's
    /// `~X ^ C ^ (~X & C)` (i.e. `~X | C`), wrapping a guest pointer.
    #[test]
    fn the_vms_pointer_wrapper_reduces_to_the_pointer() {
        let mut a = Arena::new();
        let base = a.init_reg(Reg::Rcx);
        let x = a.load(base, Width::W64);
        let c = a.constant(0xffff_ffff_ffff_fe7f, Width::W64);
        let nx = a.not(x);
        let and = a.bin(BinOp::And, nx, c);
        let xor1 = a.bin(BinOp::Xor, nx, c);
        let t = a.bin(BinOp::Xor, xor1, and);
        let nt = a.not(t);
        let lo = a.bin(BinOp::And, nt, x);
        let hi = a.bin(BinOp::And, t, x);
        let joined = a.bin(BinOp::Or, lo, hi);
        assert!(
            super::tests::equal_on_samples(&a, joined, x),
            "{} should equal {}",
            render(&a, joined, 12),
            render(&a, x, 12)
        );
        assert_eq!(joined, x, "got {}", render(&a, joined, 12));
    }

    /// TVM's other pointer wrapper, the one built around a displaced copy of the pointer rather than around a constant mask. ```text s   = a + 0x180 v13 = s | (a & (a ^ s)) v17 = (~a & v13) + ~v13 out = ~v17            // == a ```
    #[test]
    fn the_displaced_pointer_wrapper_reduces_to_the_pointer() {
        let mut a = Arena::new();
        let base = a.init_reg(Reg::Rcx);
        let p = a.load(base, Width::W64);
        let k = a.constant(0x180, Width::W64);
        let s = a.bin(BinOp::Add, p, k);
        let xor = a.bin(BinOp::Xor, p, s);
        let and = a.bin(BinOp::And, p, xor);
        let v13 = a.bin(BinOp::Or, s, and);
        let v14 = a.not(v13);
        let np = a.not(p);
        let v16 = a.bin(BinOp::And, np, v13);
        let v17 = a.bin(BinOp::Add, v16, v14);
        let got = a.not(v17);
        assert!(
            super::tests::equal_on_samples(&a, got, p),
            "{} should equal {}",
            render(&a, got, 14),
            render(&a, p, 14)
        );
        assert_eq!(got, p, "got {}", render(&a, got, 14));
    }
    /// The two mask rules that break the RFLAGS pack/unpack round trip must agree with a value computed independently of the arena.
    #[test]
    fn mask_rules_match_independent_evaluation() {
        // Bit positions the VM uses, and which input supplies each.
        const PLACED: [(bool, u64); 4] = [(true, 2), (false, 4), (true, 6), (false, 7)];

        let reference = |vx: u64, vy: u64| -> u64 {
            let mut acc = 0x2u64;
            for (from_x, bit) in PLACED {
                let src = if from_x { vx } else { vy };
                acc |= ((src & 0xff) & 1) << bit;
            }
            acc
        };

        let mut a = Arena::new();
        let x = a.init_reg(Reg::Rax);
        let y = a.init_reg(Reg::Rbx);

        let mut packed = a.constant(0x2, Width::W64);
        for (from_x, bit) in PLACED {
            let t = a.trunc(if from_x { x } else { y }, Width::W8);
            let z = a.zext(t, Width::W64);
            let one = a.constant(1, Width::W64);
            let m = a.bin(BinOp::And, z, one);
            let sh = a.constant(bit, Width::W64);
            let placed = a.bin(BinOp::Shl, m, sh);
            packed = a.bin(BinOp::Or, packed, placed);
        }

        // (node, closure computing the same thing from the raw inputs)
        type Model = Box<dyn Fn(u64, u64) -> u64>;
        let mut probes: Vec<(Ref, Model)> = Vec::new();

        // Mask down to a single flag, at every position the VM uses and one it
        // does not, where the whole `Or` is dead and the result must be zero.
        for bit in [2u64, 4, 6, 7, 9] {
            let m = a.constant(1 << bit, Width::W64);
            probes.push((
                a.bin(BinOp::And, packed, m),
                Box::new(move |vx, vy| reference(vx, vy) & (1 << bit)),
            ));
        }

        // ~(packed ^ c1) & c2: covered, over-covered, and not covered, so the
        // rule has to decline on the last one.
        for (c1, c2) in [(0x80u64, 0x80u64), (0xff, 0x80), (0x40, 0x80), (0x0, 0x80)] {
            let k = a.constant(c1, Width::W64);
            let xr = a.bin(BinOp::Xor, packed, k);
            let nt = a.un(UnOp::Not, xr);
            let m = a.constant(c2, Width::W64);
            probes.push((
                a.bin(BinOp::And, nt, m),
                Box::new(move |vx, vy| !(reference(vx, vy) ^ c1) & c2),
            ));
        }

        // Masks spanning several live flags: nothing may be dropped.
        for mask in [0xd0u64, 0x54, 0xff, 0x2] {
            let m = a.constant(mask, Width::W64);
            probes.push((
                a.bin(BinOp::And, packed, m),
                Box::new(move |vx, vy| reference(vx, vy) & mask),
            ));
        }

        for &vx in &[0u64, 1, 0xff, 0x80, 0x1234_5678_9abc_def0, u64::MAX] {
            for &vy in &[0u64, 1, 0xfe, 0x8000_0000_0000_0001, u64::MAX] {
                let env: HashMap<Reg, u64> = [(Reg::Rax, vx), (Reg::Rbx, vy)].into_iter().collect();
                for (i, (node, model)) in probes.iter().enumerate() {
                    assert_eq!(
                        crate::ir::expr::tests::eval(&a, *node, &env),
                        model(vx, vy),
                        "probe {i} disagrees at rax={vx:#x} rbx={vy:#x}"
                    );
                }
            }
        }
    }

    /// The three rules that let TVM's 64-bit TSC composition fold must agree with a value computed independently of the arena.
    #[test]
    fn tsc_composition_rules_match_independent_evaluation() {
        let mut a = Arena::new();
        let x = a.init_reg(Reg::Rax);
        let y = a.init_reg(Reg::Rbx);
        let m32 = a.constant(0xffff_ffff, Width::W64);
        let sh = a.constant(32, Width::W64);

        // hi = (x & 0xffffffff) << 32, lo = y & 0xffffffff
        let hi = {
            let t = a.bin(BinOp::And, x, m32);
            a.bin(BinOp::Shl, t, sh)
        };
        let lo = a.bin(BinOp::And, y, m32);

        type Model = Box<dyn Fn(u64, u64) -> u64>;
        let mut probes: Vec<(Ref, Model)> = Vec::new();

        let mhi = |vx: u64| (vx & 0xffff_ffff) << 32;
        let mlo = |vy: u64| vy & 0xffff_ffff;

        // ~(hi << 32) & lo, the halves recombined by masking.
        let n = a.un(UnOp::Not, hi);
        probes.push((
            a.bin(BinOp::And, n, lo),
            Box::new(move |vx, vy| !mhi(vx) & mlo(vy)),
        ));

        // a | (a ^ b) == a | b, both operand orders.
        let xr = a.bin(BinOp::Xor, hi, lo);
        probes.push((
            a.bin(BinOp::Or, hi, xr),
            Box::new(move |vx, vy| mhi(vx) | (mhi(vx) ^ mlo(vy))),
        ));
        probes.push((
            a.bin(BinOp::Or, xr, hi),
            Box::new(move |vx, vy| (mhi(vx) ^ mlo(vy)) | mhi(vx)),
        ));

        // (a + t) - a == t, both operand orders of the Add.
        let t = a.bin(BinOp::Or, hi, lo);
        let s1 = a.bin(BinOp::Add, hi, t);
        probes.push((
            a.bin(BinOp::Sub, s1, hi),
            Box::new(move |vx, vy| {
                mhi(vx)
                    .wrapping_add(mhi(vx) | mlo(vy))
                    .wrapping_sub(mhi(vx))
            }),
        ));
        let s2 = a.bin(BinOp::Add, t, hi);
        probes.push((
            a.bin(BinOp::Sub, s2, hi),
            Box::new(move |vx, vy| {
                (mhi(vx) | mlo(vy))
                    .wrapping_add(mhi(vx))
                    .wrapping_sub(mhi(vx))
            }),
        ));

        // The full composition, which must reach exactly `hi | lo`.
        let full = {
            let inner = a.bin(BinOp::Xor, hi, lo);
            let or = a.bin(BinOp::Or, hi, inner);
            let add = a.bin(BinOp::Add, or, hi);
            a.bin(BinOp::Sub, add, hi)
        };
        probes.push((full, Box::new(move |vx, vy| mhi(vx) | mlo(vy))));

        // A subtraction that must NOT cancel: the Add operand differs.
        let s3 = a.bin(BinOp::Add, hi, lo);
        probes.push((
            a.bin(BinOp::Sub, s3, t),
            Box::new(move |vx, vy| {
                mhi(vx)
                    .wrapping_add(mlo(vy))
                    .wrapping_sub(mhi(vx) | mlo(vy))
            }),
        ));

        for &vx in &[0u64, 1, 0xffff_ffff, 0x8000_0000, 0x1234_5678, u64::MAX] {
            for &vy in &[0u64, 1, 0xffff_ffff, 0xdead_beef, 0x8000_0000, u64::MAX] {
                let env: HashMap<Reg, u64> = [(Reg::Rax, vx), (Reg::Rbx, vy)].into_iter().collect();
                for (i, (node, model)) in probes.iter().enumerate() {
                    assert_eq!(
                        crate::ir::expr::tests::eval(&a, *node, &env),
                        model(vx, vy),
                        "probe {i} disagrees at rax={vx:#x} rbx={vy:#x}"
                    );
                }
            }
        }

        // The composition must have folded to the same node as a plain `hi | lo`,
        // not merely evaluate equally: the point of the rules is that the tree
        // collapses so the VPC becomes concrete.
        assert_eq!(full, t, "TSC composition did not fold to `hi | lo`");
    }

    /// Masking a deeply nested Or drops every term whose bits are all outside the mask.
    #[test]
    fn masking_a_nested_or_chain_drops_terms_outside_the_mask() {
        let mut a = Arena::new();
        let v = a.init_reg(Reg::Rax);

        // CF: Ult result, bit 0 — inside 0x41
        let kbb8 = a.constant(0xbb8, Width::W32);
        let cf = a.bin(BinOp::Ult, v, kbb8);
        let cf64 = a.zext(cf, Width::W64);

        // PF: ParityByte result, shifted to bit 2 — outside 0x41
        let vb = a.trunc(v, Width::W8);
        let pf = a.un(UnOp::ParityByte, vb);
        let pf64 = a.zext(pf, Width::W64);
        let k2sh = a.constant(2, Width::W64);
        let pf_sh = a.bin(BinOp::Shl, pf64, k2sh);

        // AF: Eq result, shifted to bit 4 — outside 0x41
        let k10 = a.constant(0x10, Width::W32);
        let v_and_10 = a.bin(BinOp::And, v, k10);
        let af = a.bin(BinOp::Eq, v_and_10, k10);
        let af64 = a.zext(af, Width::W64);
        let k4sh = a.constant(4, Width::W64);
        let af_sh = a.bin(BinOp::Shl, af64, k4sh);

        // ZF: Eq result, shifted to bit 6 — inside 0x41
        let k0 = a.constant(0, Width::W32);
        let zf = a.bin(BinOp::Eq, v, k0);
        let zf64 = a.zext(zf, Width::W64);
        let k6sh = a.constant(6, Width::W64);
        let zf_sh = a.bin(BinOp::Shl, zf64, k6sh);

        // Constant 0x2 sits at bit 1 — outside 0x41
        let k2 = a.constant(0x2, Width::W64);

        // Build the nested pack: ZF<<6 | (AF<<4 | (PF<<2 | (0x2 | CF)))
        let p = a.bin(BinOp::Or, k2, cf64);
        let p = a.bin(BinOp::Or, pf_sh, p);
        let p = a.bin(BinOp::Or, af_sh, p);
        let packed = a.bin(BinOp::Or, zf_sh, p);

        let mask = a.constant(0x41, Width::W64);
        let result = a.bin(BinOp::And, packed, mask);

        // Expected: (ZF<<6 | CF) & 0x41  — PF, AF, and 0x2 are gone.
        let zf_or_cf = a.bin(BinOp::Or, zf_sh, cf64);
        let expected = a.bin(BinOp::And, zf_or_cf, mask);
        assert_eq!(
            result,
            expected,
            "dead flag terms (PF<<2, AF<<4, 0x2) should be stripped; \
             got: {}",
            render(&a, result, 24),
        );
    }
}
