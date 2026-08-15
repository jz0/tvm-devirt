//! Unit tests for code generation.
//!
//! A child module of `codegen`, so these reach its private emitters directly.
//! `emit_one`, `text_of` and `seg` are the shared fixtures; the rest are
//! grouped by the part of codegen they cover.

use super::*;
use crate::ir::expr::BinOp;
use crate::ir::regalloc::{Item, Segment};
use crate::ir::sched::ValId;

/// Assemble one `Instr` with a hand-written allocation and disassemble it.
fn emit_one(instr: &Instr, segs: Vec<Segment>) -> Vec<String> {
    let mut asm = CodeAssembler::new(64).expect("assembler");
    let alloc = Alloc {
        segments: segs,
        copies: Vec::new(),
        slots: 0,
    };
    let layout = FrameLayout::new(0, 0, false);
    emit_instr(
        &mut asm,
        &alloc,
        instr,
        0,
        &layout,
        ImageRange::none(),
        None,
        &CallRedirects::none(),
    )
    .expect("emit");
    let bytes = asm.assemble(0x1000).expect("assemble");
    let mut dec = iced_x86::Decoder::with_ip(64, &bytes, 0x1000, 0);
    let mut fmt = iced_x86::NasmFormatter::new();
    let mut out = Vec::new();
    while dec.can_decode() {
        let insn = dec.decode();
        let mut t = String::new();
        iced_x86::Formatter::format(&mut fmt, &insn, &mut t);
        out.push(t);
    }
    out
}

/// Verify that an adjacent scratch restore/save pair is removed while the reversed
/// store/load pair is retained.
#[test]
fn adjacent_restore_then_save_of_same_slot_and_register_is_removed() {
    let base: u64 = 0x1000;
    let mut asm = CodeAssembler::new(64).expect("asm");
    // restore: mov rax, [rsp+0x108]
    asm.mov(
        iced_x86::code_asm::rax,
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + 0x108_i32),
    )
    .expect("restore");
    // save: mov [rsp+0x108], rax
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + 0x108_i32),
        iced_x86::code_asm::rax,
    )
    .expect("save");
    let bytes = asm.assemble(base).expect("assemble");
    assert_eq!(bytes.len(), 16, "two 8-byte instructions");
    let out = optimize_frame_traffic(bytes, base).expect("optimize");
    assert!(
        out.is_empty(),
        "both instructions should be gone, got {} bytes: {:?}",
        out.len(),
        disasm_bytes(&out, base)
    );
}

/// Reversed order: store then load. That is a real value transfer (slot ← rax,
/// then rax ← slot), not a no-op, and must survive the pass unchanged.
#[test]
fn a_store_followed_by_a_load_is_not_a_roundtrip_and_is_left_alone() {
    let base: u64 = 0x1000;
    let mut asm = CodeAssembler::new(64).expect("asm");
    // store first
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + 0x108_i32),
        iced_x86::code_asm::rax,
    )
    .expect("save");
    // load second
    asm.mov(
        iced_x86::code_asm::rax,
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + 0x108_i32),
    )
    .expect("restore");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic(original.clone(), base).expect("optimize");
    assert_eq!(out, original, "reversed pair must not be touched");
}

/// A restore and save at different displacements are not a removable round trip.
#[test]
fn mismatched_slot_displacements_are_not_a_roundtrip() {
    let base: u64 = 0x1000;
    let mut asm = CodeAssembler::new(64).expect("asm");
    // restore rax from slot A
    asm.mov(
        iced_x86::code_asm::rax,
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + 0x108_i32),
    )
    .expect("restore");
    // save rax to slot B (different displacement)
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + 0x110_i32),
        iced_x86::code_asm::rax,
    )
    .expect("save");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic(original.clone(), base).expect("optimize");
    assert_eq!(out, original, "different slots: must not be touched");
}

/// `mov rax,[rsp+k]` then `mov [rsp+k],rcx` is not a roundtrip: the slot ends up holding RCX, not what it held before.
#[test]
fn a_different_register_written_to_the_slot_is_not_a_roundtrip() {
    let base: u64 = 0x1000;
    let mut asm = CodeAssembler::new(64).expect("asm");
    asm.mov(
        iced_x86::code_asm::rax,
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + 0x108_i32),
    )
    .expect("restore");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + 0x108_i32),
        iced_x86::code_asm::rcx,
    )
    .expect("save");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic(original.clone(), base).expect("optimize");
    assert_eq!(out, original, "different register: must not be touched");
}

// Dead-store elimination on private frame slots

/// A layout with room for `spills` spill slots and one saved register, so the tests below have a concrete private range to aim at.
fn dse_layout(spills: u32) -> FrameLayout {
    FrameLayout::with_saves(0, spills, false, vec![Reg::Rbx])
}

/// A store to a spill slot that nothing ever reads back is removed.
#[test]
fn a_spill_store_no_one_reads_is_removed() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(1) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("store");
    asm.ret().expect("ret");
    let bytes = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(bytes, base, Some(&layout)).expect("optimize");
    let text = disasm_bytes(&out, base);
    assert_eq!(text, vec!["ret"], "dead spill store should be gone");
}

/// The same store survives when a later instruction reads the slot. The mutation this catches is a transfer function that kills the written bytes without first unioning in what the successors read: that would report every store dead, including the ones carrying a real value.
#[test]
fn a_spill_store_that_is_read_back_survives() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(1) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("store");
    asm.mov(
        iced_x86::code_asm::rcx,
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
    )
    .expect("load");
    asm.ret().expect("ret");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(out, original, "a store whose value is read must survive");
}

/// A store to the callee-saved register area is never removed, even though no instruction in the function reads it back. The OS unwinder restores `rbx` from that slot by following the unwind record, which is not visible in the instruction stream at all.
#[test]
fn a_store_to_the_callee_saved_area_is_never_removed() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let save = layout.save_disp(Reg::Rbx).expect("rbx is saved") as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + save),
        iced_x86::code_asm::rbx,
    )
    .expect("save rbx");
    asm.ret().expect("ret");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(
        out, original,
        "the unwinder reads this slot; it is not dead"
    );
}

/// A store to a guest local is never removed. `Frame(o)` slots can have their address taken and passed to a callee, so "no instruction in this function reads it" does not imply the store is unobservable.
#[test]
fn a_store_to_a_guest_local_is_never_removed() {
    let base: u64 = 0x1000;
    // frame_lo of -0x40 gives the layout 0x40 bytes of guest locals.
    let layout = FrameLayout::with_saves(-0x40, 2, false, vec![Reg::Rbx]);
    let local = layout.frame_disp(-0x20) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + local),
        iced_x86::code_asm::rax,
    )
    .expect("store local");
    asm.ret().expect("ret");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(out, original, "guest locals may be read through a pointer");
}

/// A 32-bit store leaves the slot's upper half alone, so it cannot be removed on the strength of a 64-bit read of the same slot being satisfied elsewhere.
#[test]
fn a_narrow_store_does_not_kill_the_whole_slot() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(1) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    // Wide store supplies all eight bytes.
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("wide store");
    // Narrow store overwrites only the low four.
    asm.mov(
        iced_x86::code_asm::dword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::ecx,
    )
    .expect("narrow store");
    // Wide read needs the upper four from the first store.
    asm.mov(
        iced_x86::code_asm::rdx,
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
    )
    .expect("wide load");
    asm.ret().expect("ret");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(
        out, original,
        "the wide store still supplies the upper half"
    );
}

/// A store whose value is read only on one side of a branch survives.
///
/// The liveness merge at a branch is a union over successors. Intersecting instead
/// would call this store dead because the fall-through path does not read it.
#[test]
fn a_store_read_on_only_one_branch_path_survives() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(2) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    let mut reader = asm.create_label();
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("store");
    asm.je(reader).expect("branch");
    // Fall-through path: never touches the slot.
    asm.ret().expect("ret");
    asm.set_label(&mut reader).expect("label");
    asm.mov(
        iced_x86::code_asm::rcx,
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
    )
    .expect("load");
    asm.ret().expect("ret");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(
        out, original,
        "one reading path is enough to keep the store"
    );
}

/// A store read back only after going round a loop survives. The backward pass is iterated to a fixpoint because of this shape: on the first sweep the backward edge's target has an empty live-in set, so the store looks dead.
#[test]
fn a_store_read_across_a_backward_edge_survives() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(3) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    let mut top = asm.create_label();
    asm.set_label(&mut top).expect("label");
    // Read first, then store: the read is satisfied by the store from the
    // *previous* iteration, which only a fixpoint discovers.
    asm.mov(
        iced_x86::code_asm::rcx,
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
    )
    .expect("load");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("store");
    asm.jne(top).expect("loop");
    asm.ret().expect("ret");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(out, original, "the loop's next iteration reads this store");
}

/// An indirect branch disables dead-store elimination entirely.
///
/// The target is not known here, so a slot stored before the jump may be read at
/// the destination. Bailing is the only sound answer; the pass returns the input
/// untouched rather than guessing.
#[test]
fn an_indirect_branch_disables_dead_store_elimination() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(1) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("store");
    asm.jmp(iced_x86::code_asm::rcx).expect("indirect jmp");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(
        out, original,
        "an unresolved indirect branch must disable the pass"
    );
}

/// Taking the address of a private slot disables dead-store elimination. The pass's premise is that spill and scratch slots are reached only through the direct `[rsp+k]` operands this emitter writes.
#[test]
fn taking_the_address_of_a_private_slot_disables_dead_store_elimination() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(1) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("store");
    asm.lea(
        iced_x86::code_asm::rcx,
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
    )
    .expect("lea");
    asm.ret().expect("ret");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(
        out, original,
        "an address-taken private slot must disable the pass"
    );
}

/// A read-modify-write of a spill slot both reads and writes it, so it is never treated as a removable store. Only `mov [rsp+k], src` is classified as a pure store.
#[test]
fn a_read_modify_write_of_a_spill_slot_is_not_a_removable_store() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(1) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    asm.add(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("rmw");
    asm.ret().expect("ret");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(out, original, "an rmw is a read as well as a write");
}

/// Branch targets remain valid when preceding instructions are removed.
#[test]
fn branch_targets_survive_compaction() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let dead_slot = layout.spill_disp(1) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    let mut target = asm.create_label();
    asm.je(target).expect("branch");
    // Two dead stores between the branch and its target.
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + dead_slot),
        iced_x86::code_asm::rax,
    )
    .expect("dead 1");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + dead_slot),
        iced_x86::code_asm::rcx,
    )
    .expect("dead 2");
    asm.set_label(&mut target).expect("label");
    asm.xor(iced_x86::code_asm::eax, iced_x86::code_asm::eax)
        .expect("marker");
    asm.ret().expect("ret");
    let bytes = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(bytes, base, Some(&layout)).expect("optimize");
    let text = disasm_bytes(&out, base);
    assert_eq!(
        text,
        vec!["je short 0000000000001002h", "xor eax,eax", "ret"],
        "the branch should now target the compacted position of `xor`"
    );
}

/// A dead store that is a branch target is retained.
#[test]
fn a_dead_store_at_a_branch_target_is_left_in_place() {
    let base: u64 = 0x1000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(1) as i32;
    let mut asm = CodeAssembler::new(64).expect("asm");
    let mut target = asm.create_label();
    asm.je(target).expect("branch");
    asm.nop().expect("filler");
    asm.set_label(&mut target).expect("label");
    // Dead by liveness, but reachable directly from the branch above.
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("dead store at target");
    asm.ret().expect("ret");
    let original = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(original.clone(), base, Some(&layout)).expect("optimize");
    assert_eq!(out, original, "a branch target must keep its instruction");
}

/// A call out of the function keeps its absolute target after compaction. `BlockEncoder` re-resolves branches by matching target IPs against the instructions in the block.
#[test]
fn an_external_call_target_is_unchanged_by_compaction() {
    let base: u64 = 0x140001000;
    let layout = dse_layout(4);
    let slot = layout.spill_disp(1) as i32;
    let target: u64 = 0x140099000;
    let mut asm = CodeAssembler::new(64).expect("asm");
    asm.mov(
        iced_x86::code_asm::qword_ptr(iced_x86::code_asm::rsp + slot),
        iced_x86::code_asm::rax,
    )
    .expect("dead store");
    asm.call(target).expect("call");
    asm.ret().expect("ret");
    let bytes = asm.assemble(base).expect("assemble");
    let out = optimize_frame_traffic_with(bytes, base, Some(&layout)).expect("optimize");
    let text = disasm_bytes(&out, base);
    assert_eq!(
        text,
        vec!["call 0000000140099000h".to_string(), "ret".to_string()],
        "the external call must still reach the same address"
    );
}

// Call redirection through retargeted trampolines

/// A direct call to a retargeted trampoline names the devirtualized body instead. Recovery folds a call to a virtualized function to that function's *trampoline*, which `write_devirt` overwrites with `jmp <body>`.
#[test]
fn a_direct_call_to_a_retargeted_trampoline_names_the_body() {
    let layout = FrameLayout::new(0, 0, true);
    let alloc = Alloc {
        segments: Vec::new(),
        copies: Vec::new(),
        slots: 0,
    };
    let instr = Instr::Call {
        target: CallTarget::Direct(0x140002bec),
        args: Vec::new(),
    };
    let mut redirect = CallRedirects::none();
    redirect.insert(0x140002bec, 0x1402c1370);

    let mut asm = CodeAssembler::new(64).expect("asm");
    emit_instr(
        &mut asm,
        &alloc,
        &instr,
        0,
        &layout,
        ImageRange::none(),
        None,
        &redirect,
    )
    .expect("emit");
    let bytes = asm.assemble(0x1402c4fb8).expect("assemble");
    let text = disasm_bytes(&bytes, 0x1402c4fb8);
    assert_eq!(
        text,
        vec!["call 00000001402C1370h"],
        "the call should reach the body, not the trampoline"
    );
}

/// A direct call to an address with no redirect is left exactly as recovered. Functions that were never virtualized, and those whose recovery did not finish and so keep their original trampoline, must still be called at the address recovery produced.
#[test]
fn a_direct_call_with_no_redirect_is_unchanged() {
    let layout = FrameLayout::new(0, 0, true);
    let alloc = Alloc {
        segments: Vec::new(),
        copies: Vec::new(),
        slots: 0,
    };
    let instr = Instr::Call {
        target: CallTarget::Direct(0x140027980),
        args: Vec::new(),
    };
    // A populated map that does not mention this target.
    let mut redirect = CallRedirects::none();
    redirect.insert(0x140002bec, 0x1402c1370);

    let mut asm = CodeAssembler::new(64).expect("asm");
    emit_instr(
        &mut asm,
        &alloc,
        &instr,
        0,
        &layout,
        ImageRange::none(),
        None,
        &redirect,
    )
    .expect("emit");
    let bytes = asm.assemble(0x1402c4f8d).expect("assemble");
    let text = disasm_bytes(&bytes, 0x1402c4f8d);
    assert_eq!(text, vec!["call 0000000140027980h"]);
}

/// A tail call through a retargeted trampoline is redirected too. `jmp rel32` has the same hop problem and the same five-byte encoding, so it gets the same treatment.
#[test]
fn a_tail_call_to_a_retargeted_trampoline_names_the_body() {
    let layout = FrameLayout::new(0, 0, false);
    let alloc = Alloc {
        segments: Vec::new(),
        copies: Vec::new(),
        slots: 0,
    };
    let b = SchedBlock {
        id: BlockId {
            handler: 0x1000,
            vip: None,
        },
        instrs: Vec::new(),
        exits: Vec::new(),
        callee_set: Vec::new(),
        control: None,
        terminator: Terminator::TailCall {
            target: 0x140002bec,
        },
        fused_cmp: None,
    };
    let label_of = HashMap::new();
    let mut redirect = CallRedirects::none();
    redirect.insert(0x140002bec, 0x1402c1370);

    let mut asm = CodeAssembler::new(64).expect("asm");
    emit_terminator(
        &mut asm,
        &alloc,
        &b,
        &[],
        &label_of,
        &layout,
        None,
        &redirect,
    )
    .expect("emit");
    let bytes = asm.assemble(0x1402c6000).expect("assemble");
    let text = disasm_bytes(&bytes, 0x1402c6000);
    assert!(
        text.iter().any(|t| t == "jmp 00000001402C1370h"),
        "tail call should reach the body, got {text:?}"
    );
}

fn seg(item: Item, reg: Reg) -> Segment {
    Segment {
        item,
        start: 0,
        end: 8,
        loc: Loc::Reg(reg),
    }
}

/// Disassemble raw assembled bytes.
///
/// Distinct from [`text_of`], which assembles a whole `Instr` stream itself. The
/// relocation tests below need to inspect the bytes of a hand-built emit, so they
/// assemble first and read back here.
fn disasm_bytes(bytes: &[u8], base: u64) -> Vec<String> {
    let mut dec = iced_x86::Decoder::with_ip(64, bytes, base, 0);
    let mut fmt = iced_x86::NasmFormatter::new();
    let mut out = Vec::new();
    while dec.can_decode() {
        let insn = dec.decode();
        let mut t = String::new();
        iced_x86::Formatter::format(&mut fmt, &insn, &mut t);
        out.push(t);
    }
    out
}

#[test]
fn a_select_emits_a_conditional_move_for_its_false_operand() {
    let instr = Instr::Select {
        dst: ValId(0),
        cond: Operand::Val(ValId(1)),
        a: Operand::Val(ValId(2)),
        b: Operand::Val(ValId(3)),
        width: Width::W64,
    };
    let text = emit_one(
        &instr,
        vec![
            seg(Item::Val(ValId(0)), Reg::Rdx),
            seg(Item::Val(ValId(1)), Reg::Rcx),
            seg(Item::Val(ValId(2)), Reg::Rdx),
            seg(Item::Val(ValId(3)), Reg::R8),
        ],
    );
    let joined = text.join("; ");
    assert!(
        text.iter().any(|t| t.starts_with("cmove")),
        "the false operand needs a conditional move, got: {joined}"
    );
    assert!(
        text.iter().any(|t| t.contains("cmove rdx,r8")),
        "cmovz must take `b` into `dst`, got: {joined}"
    );
    // An 8-bit predicate must be tested 8-bit: the bytes above it were never
    // defined by whatever produced the flag.
    assert!(
        text.iter().any(|t| t == "test cl,cl"),
        "the predicate is a W8 value, got: {joined}"
    );
}

/// A W32 `Select` uses the 32-bit CMOV form, which also clears the upper half.
#[test]
fn a_32_bit_select_uses_the_32_bit_conditional_move() {
    let instr = Instr::Select {
        dst: ValId(0),
        cond: Operand::Val(ValId(1)),
        a: Operand::Val(ValId(2)),
        b: Operand::Val(ValId(3)),
        width: Width::W32,
    };
    let text = emit_one(
        &instr,
        vec![
            seg(Item::Val(ValId(0)), Reg::Rdx),
            seg(Item::Val(ValId(1)), Reg::Rcx),
            seg(Item::Val(ValId(2)), Reg::Rdx),
            seg(Item::Val(ValId(3)), Reg::R8),
        ],
    );
    let joined = text.join("; ");
    assert!(
        text.iter().any(|t| t.contains("cmove edx,r8d")),
        "expected a 32-bit cmove, got: {joined}"
    );
}

#[test]
fn a_16_bit_store_emits_a_16_bit_move() {
    let instr = Instr::Store {
        addr: Operand::Frame(-8),
        value: Operand::Val(ValId(1)),
        width: Width::W16,
        disp: 0,
    };
    let text = emit_one(&instr, vec![seg(Item::Val(ValId(1)), Reg::Rdx)]);
    let joined = text.join("; ");
    assert!(!text.is_empty(), "a 16-bit store must emit something");
    // NasmFormatter omits the size when the register operand implies it, so the
    // 16-bit form shows as `mov [rsp],dx`; `dx`, not `edx`, is the assertion.
    assert!(
        text.iter().any(|t| t.contains(",dx")),
        "expected a word-sized store of dx, got: {joined}"
    );
    assert!(
        !text.iter().any(|t| t.contains("edx") || t.contains("rdx")),
        "a W16 store must not widen past dx, got: {joined}"
    );
}

/// The same, for a 16-bit immediate.
#[test]
fn a_16_bit_immediate_store_emits_a_word_move() {
    let instr = Instr::Store {
        addr: Operand::Frame(-8),
        value: Operand::Imm(0x6b),
        width: Width::W16,
        disp: 0,
    };
    let text = emit_one(&instr, Vec::new());
    let joined = text.join("; ");
    assert!(
        text.iter()
            .any(|t| t.contains("word [") && !t.contains("dword [")),
        "expected a word-sized immediate store, got: {joined}"
    );
}

/// A wide immediate image address is materialized RIP-relatively before the store.
#[test]
fn a_wide_immediate_store_of_an_image_address_uses_a_rip_relative_lea() {
    let image = ImageRange {
        lo: 0x1_4000_0000,
        hi: 0x1_4040_0000,
    };
    let target = 0x1_4000_1690_u64;
    assert!(
        (target as i64) > i32::MAX as i64,
        "premise: too wide for the imm32 store form, so it must go via a register"
    );
    let instr = Instr::Store {
        addr: Operand::Val(ValId(1)),
        value: Operand::Imm(target),
        width: Width::W64,
        disp: 0x70,
    };
    let mut asm = CodeAssembler::new(64).expect("assembler");
    let alloc = Alloc {
        segments: vec![seg(Item::Val(ValId(1)), Reg::Rcx)],
        copies: Vec::new(),
        slots: 0,
    };
    let mut layout = FrameLayout::new(0, 0, false);
    layout.img = image;
    emit_instr(
        &mut asm,
        &alloc,
        &instr,
        0,
        &layout,
        image,
        None,
        &CallRedirects::none(),
    )
    .expect("emit");
    // Assemble at a base inside the image, as real emission does: a RIP-relative
    // operand is only encodable within +/-2GB of the instruction.
    let base = 0x1_4018_930c_u64;
    let bytes = asm.assemble(base).expect("assemble");
    let text = disasm_bytes(&bytes, base);
    let joined = text.join("; ");
    assert!(
        text.iter()
            .any(|t| t.starts_with("lea ") && t.contains("rel ")),
        "the address must be computed RIP-relative, got: {joined}"
    );
    assert!(
        !text
            .iter()
            .any(|t| t.contains(&format!("{target:X}h")) && t.starts_with("mov ")),
        "no absolute image immediate may survive, got: {joined}"
    );
}

/// The same guard for a numeric constant *outside* the image: it stays an immediate. The RIP-relative rewrite is only correct for addresses.
#[test]
fn a_wide_immediate_store_outside_the_image_stays_an_immediate() {
    let image = ImageRange {
        lo: 0x1_4000_0000,
        hi: 0x1_4040_0000,
    };
    let value = 0xdead_beef_feed_face_u64;
    assert!(!image.rip_ok(value), "premise: not an image address");
    let instr = Instr::Store {
        addr: Operand::Val(ValId(1)),
        value: Operand::Imm(value),
        width: Width::W64,
        disp: 0,
    };
    let mut asm = CodeAssembler::new(64).expect("assembler");
    let alloc = Alloc {
        segments: vec![seg(Item::Val(ValId(1)), Reg::Rcx)],
        copies: Vec::new(),
        slots: 0,
    };
    let mut layout = FrameLayout::new(0, 0, false);
    layout.img = image;
    emit_instr(
        &mut asm,
        &alloc,
        &instr,
        0,
        &layout,
        image,
        None,
        &CallRedirects::none(),
    )
    .expect("emit");
    let bytes = asm.assemble(0x1000).expect("assemble");
    let text = disasm_bytes(&bytes, 0x1000);
    let joined = text.join("; ");
    assert!(
        text.iter().any(|t| t.contains("DEADBEEFFEEDFACE")),
        "a non-address constant must be stored as itself, got: {joined}"
    );
    assert!(
        !text.iter().any(|t| t.starts_with("lea ")),
        "a non-address constant must not be relocated, got: {joined}"
    );
}

#[test]
fn a_fused_comparison_uses_the_operand_width() {
    let v = ValId(0);
    // Only the fused form is consulted for the condition, but a `Branch` needs a
    // predicate `Ref`, so make a real one.
    let mut arena = crate::ir::expr::Arena::new();
    let predicate = arena.constant(1, Width::W8);
    let mut b = SchedBlock {
        id: BlockId {
            handler: 0x1000,
            vip: None,
        },
        instrs: Vec::new(),
        exits: Vec::new(),
        callee_set: Vec::new(),
        control: Some(Operand::Val(v)),
        terminator: Terminator::Branch {
            predicate,
            taken: BlockId {
                handler: 0x2000,
                vip: None,
            },
            not_taken: BlockId {
                handler: 0x3000,
                vip: None,
            },
        },
        // Populated by `eliminate_dead` through `fusable_cmp`.
        fused_cmp: None,
    };
    let and = ValId(1);
    b.instrs.push(Instr::Bin {
        dst: and,
        op: crate::ir::expr::BinOp::And,
        a: Operand::Val(ValId(2)),
        b: Operand::Imm(0x8000_0000),
        width: Width::W32,
        operand_width: Width::W32,
    });
    // `status & 0x80000000 == 0x80000000`. The result is a boolean, so W8, but the
    // operands are W32 and that is what the `cmp` must measure.
    b.instrs.push(Instr::Bin {
        dst: v,
        op: crate::ir::expr::BinOp::Eq,
        a: Operand::Val(and),
        b: Operand::Imm(0x8000_0000),
        width: Width::W8,
        operand_width: Width::W32,
    });
    crate::ir::sched::eliminate_dead(&mut b);
    assert!(
        b.fused_cmp.is_some(),
        "the comparison should have been fused"
    );

    let mut asm = CodeAssembler::new(64).expect("assembler");
    let alloc = Alloc {
        segments: vec![
            Segment {
                item: Item::Val(and),
                start: 0,
                end: 8,
                loc: Loc::Reg(Reg::Rax),
            },
            Segment {
                item: Item::Val(v),
                start: 0,
                end: 8,
                loc: Loc::Reg(Reg::Rax),
            },
            // `rewrite_masked_sign_tests` rewrites the masked equality into a signed
            // comparison on `ValId(2)`, so that value needs an allocation.
            Segment {
                item: Item::Val(ValId(2)),
                start: 0,
                end: 8,
                loc: Loc::Reg(Reg::Rax),
            },
        ],
        copies: Vec::new(),
        slots: 0,
    };
    let layout = FrameLayout::new(0, 0, false);
    let mut label_of = HashMap::new();
    let mut taken = asm.create_label();
    let mut not_taken = asm.create_label();
    label_of.insert(
        BlockId {
            handler: 0x2000,
            vip: None,
        },
        taken,
    );
    label_of.insert(
        BlockId {
            handler: 0x3000,
            vip: None,
        },
        not_taken,
    );
    emit_terminator(
        &mut asm,
        &alloc,
        &b,
        &[],
        &label_of,
        &layout,
        None,
        &CallRedirects::none(),
    )
    .expect("emit");
    asm.set_label(&mut taken).expect("label");
    asm.nop().expect("nop");
    asm.set_label(&mut not_taken).expect("label");
    asm.nop().expect("nop");
    let bytes = asm.assemble(0x1000).expect("assemble");
    let mut dec = iced_x86::Decoder::with_ip(64, &bytes, 0x1000, 0);
    let mut fmt = iced_x86::NasmFormatter::new();
    let mut text = Vec::new();
    while dec.can_decode() {
        let insn = dec.decode();
        let mut t = String::new();
        iced_x86::Formatter::format(&mut fmt, &insn, &mut t);
        text.push(t);
    }
    // Either form is acceptable: a comparison against zero lowers to
    // `test r,r`, which sets ZF and SF from the same value. What matters is the
    // *width* of the register named.
    let cmp = text
        .iter()
        .find(|t| t.starts_with("cmp") || t.starts_with("test"))
        .unwrap_or_else(|| panic!("no comparison in {text:?}"));
    assert!(
        cmp.contains("eax"),
        "the comparison must be 32-bit; comparing the low byte makes an NTSTATUS              sign test unconditional. got {cmp:?}"
    );
    assert!(
        !cmp.contains(" al,"),
        "the comparison must not narrow to al. got {cmp:?}"
    );
}

/// Boundary moves must not overwrite registers read by the branch comparison.
#[test]
fn a_boundary_move_does_not_clobber_the_compared_register() {
    let v = ValId(0);
    let loaded = ValId(1);
    let mut arena = crate::ir::expr::Arena::new();
    let predicate = arena.constant(1, Width::W8);
    let taken_id = BlockId {
        handler: 0x2000,
        vip: None,
    };
    let not_taken_id = BlockId {
        handler: 0x3000,
        vip: None,
    };
    let mut b = SchedBlock {
        id: BlockId {
            handler: 0x1000,
            vip: None,
        },
        instrs: Vec::new(),
        // RAX leaves the block holding a stack address, exactly as the guest's
        // boolean return value did.
        exits: vec![(Reg::Rax, Operand::Frame(0))],
        callee_set: Vec::new(),
        control: Some(Operand::Val(v)),
        terminator: Terminator::Branch {
            predicate,
            taken: taken_id,
            not_taken: not_taken_id,
        },
        fused_cmp: None,
    };
    b.instrs.push(Instr::Load {
        dst: loaded,
        addr: Operand::Imm(0x140042298),
        disp: 0,
        width: Width::W64,
    });
    // `loaded == 0`, the null check.
    b.instrs.push(Instr::Bin {
        dst: v,
        op: crate::ir::expr::BinOp::Eq,
        a: Operand::Val(loaded),
        b: Operand::Imm(0),
        width: Width::W8,
        operand_width: Width::W64,
    });
    crate::ir::sched::eliminate_dead(&mut b);
    assert!(
        b.fused_cmp.is_some(),
        "the comparison should have been fused"
    );

    // The allocator put the loaded value in RAX, which is also RAX's exit
    // register. That is legal: the move is supposed to happen after the branch
    // reads it.
    let alloc = Alloc {
        segments: vec![Segment {
            item: Item::Val(loaded),
            start: 0,
            end: 8,
            loc: Loc::Reg(Reg::Rax),
        }],
        copies: Vec::new(),
        slots: 0,
    };
    let layout = FrameLayout::new(-0x30, 0, true);
    let moves = vec![Move::Frame {
        dst: Reg::Rax,
        offset: 0,
    }];

    let mut asm = CodeAssembler::new(64).expect("assembler");
    let mut label_of = HashMap::new();
    let mut taken = asm.create_label();
    let mut not_taken = asm.create_label();
    label_of.insert(taken_id, taken);
    label_of.insert(not_taken_id, not_taken);
    emit_terminator(
        &mut asm,
        &alloc,
        &b,
        &moves,
        &label_of,
        &layout,
        None,
        &CallRedirects::none(),
    )
    .expect("emit");
    asm.set_label(&mut taken).expect("label");
    asm.nop().expect("nop");
    asm.set_label(&mut not_taken).expect("label");
    asm.nop().expect("nop");
    let bytes = asm.assemble(0x1000).expect("assemble");
    let mut dec = iced_x86::Decoder::with_ip(64, &bytes, 0x1000, 0);
    let mut fmt = iced_x86::NasmFormatter::new();
    let mut text = Vec::new();
    while dec.can_decode() {
        let insn = dec.decode();
        let mut t = String::new();
        iced_x86::Formatter::format(&mut fmt, &insn, &mut t);
        text.push(t);
    }

    let cmp_at = text
        .iter()
        .position(|t| t.starts_with("cmp") || t.starts_with("test"))
        .unwrap_or_else(|| panic!("no comparison in {text:?}"));
    let lea_at = text
        .iter()
        .position(|t| t.starts_with("lea"))
        .unwrap_or_else(|| panic!("no boundary move in {text:?}"));
    assert!(
        cmp_at < lea_at,
        "the boundary move into RAX must come after the comparison reads it, \
         or the guard tests a stack address. got {text:?}"
    );
}

/// The scratch slot must not overlap anything else in the frame.
#[test]
fn the_scratch_slot_collides_with_nothing() {
    let saves = vec![Reg::Rbx, Reg::Rbp, Reg::Rdi];
    // Shapes that move each region independently: spills, saves, stack arguments,
    // locals, and the with/without-calls shadow space.
    for spills in [0u32, 1, 3, 49] {
        for stack_args in [0u32, 3] {
            for has_calls in [false, true] {
                for frame_lo in [0i64, -0x30, -0x340] {
                    let l =
                        FrameLayout::full(frame_lo, spills, has_calls, saves.clone(), stack_args);
                    let scratch = l.scratch_disp();

                    for i in 0..spills {
                        assert_ne!(
                            scratch,
                            l.spill_disp(i),
                            "scratch aliases spill slot {i} (spills={spills} \
                             args={stack_args} calls={has_calls})"
                        );
                    }
                    for r in &saves {
                        assert_ne!(
                            scratch,
                            l.save_disp(*r).expect("save placed"),
                            "scratch aliases the save slot for {r:?} (spills={spills} \
                             args={stack_args} calls={has_calls})"
                        );
                    }
                    // Outgoing arguments are addressed as `rsp+n` directly, so the
                    // test is that scratch sits clear of the whole region a callee
                    // may write: its shadow space plus our stack arguments.
                    let outgoing_top = if has_calls { 0x20 } else { 0 } + 8 * stack_args as i64;
                    assert!(
                        scratch >= outgoing_top,
                        "scratch at {scratch:#x} is inside the outgoing area                              (top {outgoing_top:#x}), which a callee may overwrite"
                    );
                    // And it has to be inside the frame the prologue reserves,
                    // with a full qword to spare.
                    assert!(
                        scratch >= 0 && scratch + 8 <= l.frame_size as i64,
                        "scratch at {scratch:#x} escapes a {:#x}-byte frame",
                        l.frame_size
                    );
                }
            }
        }
    }
}

#[test]
fn w16_binary_ops_emit_the_operation() {
    let instr = Instr::Bin {
        dst: ValId(1),
        op: crate::ir::expr::BinOp::And,
        a: Operand::Val(ValId(0)),
        b: Operand::Imm(0x8bc8),
        width: Width::W16,
        operand_width: Width::W16,
    };
    let text = emit_one(
        &instr,
        vec![
            seg(Item::Val(ValId(0)), Reg::Rax),
            seg(Item::Val(ValId(1)), Reg::Rcx),
        ],
    );
    assert!(
        text.iter().any(|t| t.contains("and") && t.contains("cx")),
        "expected a 16-bit `and`, got {text:?}"
    );
}

/// A 64-bit immediate wider than imm32 must not be truncated.
#[test]
fn wide_immediates_are_not_truncated() {
    let instr = Instr::Bin {
        dst: ValId(1),
        op: crate::ir::expr::BinOp::Add,
        a: Operand::Val(ValId(0)),
        b: Operand::Imm(0x1_4008_1ae4),
        width: Width::W64,
        operand_width: Width::W64,
    };
    let text = emit_one(
        &instr,
        vec![
            seg(Item::Val(ValId(0)), Reg::Rax),
            seg(Item::Val(ValId(1)), Reg::Rcx),
        ],
    );
    let joined = text.join(" ; ");
    assert!(
        joined.contains("140081AE4") || joined.contains("140081ae4"),
        "the full 64-bit constant must appear, got {joined}"
    );
    assert!(
        !joined.contains("40081AE4h,") && !joined.contains("add rcx,40081AE4"),
        "the truncated form must not be emitted, got {joined}"
    );
}

/// A widening cast must emit a widening instruction. `emit_cast` ended each of its `Zext`/`Sext` arms with `_ if src_r == dst_r => {}` followed by `_ => {}`, which swallowed every form except the two W8 ones and left an unreachable `_ => asm.mov(..)` below.
#[test]
fn sext32_to_64_emits_movsxd() {
    let instr = Instr::Cast {
        dst: ValId(1),
        kind: crate::ir::sched::CastKind::Sext,
        a: Operand::Val(ValId(0)),
        from: Width::W32,
        to: Width::W64,
    };
    let text = emit_one(
        &instr,
        vec![
            seg(Item::Val(ValId(0)), Reg::Rax),
            seg(Item::Val(ValId(1)), Reg::Rcx),
        ],
    );
    assert!(
        text.iter().any(|t| t.contains("movsxd")),
        "expected `movsxd`, got {text:?}"
    );
}

/// `ParityByte` must emit something: every one of its results is consumed.
#[test]
fn parity_byte_is_emitted() {
    let instr = Instr::Un {
        dst: ValId(1),
        op: crate::ir::expr::UnOp::ParityByte,
        a: Operand::Val(ValId(0)),
        width: Width::W8,
    };
    let text = emit_one(
        &instr,
        vec![
            seg(Item::Val(ValId(0)), Reg::Rax),
            seg(Item::Val(ValId(1)), Reg::Rcx),
        ],
    );
    assert!(
        text.iter().any(|t| t.contains("setp")),
        "expected `setp` to read PF back out, got {text:?}"
    );
}

#[test]
fn frame_add_does_not_clobber_the_live_index() {
    let instr = Instr::Bin {
        dst: ValId(1),
        op: crate::ir::expr::BinOp::Add,
        a: Operand::Val(ValId(0)),
        b: Operand::Frame(-0x268),
        width: Width::W64,
        operand_width: Width::W64,
    };
    // v0 (the index) is in RCX and stays live; the destination is RAX.
    let text = emit_one(
        &instr,
        vec![
            seg(Item::Val(ValId(0)), Reg::Rcx),
            seg(Item::Val(ValId(1)), Reg::Rax),
        ],
    );
    let joined = text.join(" ; ");
    assert!(
        !joined.contains("lea rcx"),
        "the live index in RCX must not be overwritten, got {joined}"
    );
    assert!(
        joined.contains("lea rax") && joined.contains("add rax,rcx"),
        "expected the address built in the destination, got {joined}"
    );
}

/// The save area is disjoint from shadow space, spills, and guest locals.
#[test]
fn save_area_is_disjoint() {
    let saves = vec![Reg::Rbx, Reg::Rdi, Reg::R12];
    for &locals in &[0i64, -8, -64, -2184] {
        for spills in [0u32, 1, 20] {
            let l = FrameLayout::with_saves(locals, spills, true, saves.clone());
            // Saves sit above the shadow space and the spill slots.
            let first = l.save_disp(Reg::Rbx).unwrap();
            assert!(first >= SHADOW + 8 * spills as i64, "save overlaps spills");
            // Every save slot is distinct and 8 bytes apart.
            let mut slots: Vec<i64> = saves.iter().map(|&r| l.save_disp(r).unwrap()).collect();
            slots.sort();
            slots.dedup();
            assert_eq!(slots.len(), saves.len(), "save slots collide");
            // The last save must end at or below the lowest guest local.
            let last = slots[slots.len() - 1] + 8;
            let lowest_local = l.frame_disp(locals.min(0));
            assert!(
                last <= lowest_local,
                "save {last} overlaps local {lowest_local}"
            );
            // And the whole frame must stay 16-byte aligned after the sub.
            assert_eq!(l.frame_size % 16, 8, "frame misaligned");
        }
    }
}

/// Outgoing stack arguments do not overlap spill or save slots.
#[test]
fn stack_args_do_not_alias_spills_or_saves() {
    let saves = vec![Reg::Rbx, Reg::R12];
    for extra in 1..=4u32 {
        let l = FrameLayout::full(-64, 3, true, saves.clone(), extra);
        let mut used: Vec<i64> = Vec::new();
        for n in 5..5 + extra {
            let d = l.stack_arg_disp(n).expect("reserved argument has a slot");
            assert!(d >= SHADOW, "argument {n} overlaps shadow space");
            used.push(d);
        }
        for i in 0..3 {
            used.push(l.spill_disp(i));
        }
        for &r in &saves {
            used.push(l.save_disp(r).unwrap());
        }
        let n = used.len();
        used.sort();
        used.dedup();
        assert_eq!(used.len(), n, "frame slots alias with {extra} stack args");
    }
}
/// A fused comparison on an entry register keeps that register live through the
/// branch.
#[test]
fn a_fused_comparison_on_an_entry_register_names_that_register() {
    let mut arena = crate::ir::expr::Arena::new();
    let predicate = arena.constant(1, Width::W8);
    let mut b = SchedBlock {
        id: BlockId {
            handler: 0x1000,
            vip: None,
        },
        instrs: vec![
            // Two ordinary values so the allocator has something to place, and
            // a reason to reuse a register if a range ends early.
            Instr::Bin {
                dst: ValId(0),
                op: crate::ir::expr::BinOp::And,
                a: Operand::Entry(Reg::Rax),
                b: Operand::Imm(1),
                width: Width::W64,
                operand_width: Width::W64,
            },
            Instr::Bin {
                dst: ValId(1),
                op: crate::ir::expr::BinOp::Shl,
                a: Operand::Val(ValId(0)),
                b: Operand::Imm(4),
                width: Width::W64,
                operand_width: Width::W64,
            },
            // The control definer: `rcx == 0`, on the entry value of rcx.
            Instr::Bin {
                dst: ValId(2),
                op: crate::ir::expr::BinOp::Eq,
                a: Operand::Entry(Reg::Rcx),
                b: Operand::Imm(0),
                width: Width::W8,
                operand_width: Width::W64,
            },
        ],
        exits: vec![(Reg::Rbx, Operand::Val(ValId(1)))],
        callee_set: Vec::new(),
        control: Some(Operand::Val(ValId(2))),
        terminator: Terminator::Branch {
            predicate,
            taken: BlockId {
                handler: 0x2000,
                vip: None,
            },
            not_taken: BlockId {
                handler: 0x3000,
                vip: None,
            },
        },
        fused_cmp: None,
    };

    crate::ir::sched::eliminate_dead(&mut b);
    assert!(
        b.fused_cmp.is_some(),
        "the comparison should have been fused"
    );
    assert!(
        matches!(&b.fused_cmp, Some(fc) if fc.a == Operand::Entry(Reg::Rcx)),
        "the fused comparison should read rcx's entry value, got {:?}",
        b.fused_cmp
    );

    let alloc = regalloc::allocate(&b);
    let mut asm = CodeAssembler::new(64).expect("assembler");
    let layout = FrameLayout::new(0, 0, false);
    let mut label_of = HashMap::new();
    let mut taken = asm.create_label();
    let mut not_taken = asm.create_label();
    label_of.insert(
        BlockId {
            handler: 0x2000,
            vip: None,
        },
        taken,
    );
    label_of.insert(
        BlockId {
            handler: 0x3000,
            vip: None,
        },
        not_taken,
    );
    emit_terminator(
        &mut asm,
        &alloc,
        &b,
        &[],
        &label_of,
        &layout,
        None,
        &CallRedirects::none(),
    )
    .expect("emit must not fail: rcx's entry value has a location");
    asm.set_label(&mut taken).expect("label");
    asm.nop().expect("nop");
    asm.set_label(&mut not_taken).expect("label");
    asm.nop().expect("nop");

    let bytes = asm.assemble(0x1000).expect("assemble");
    let mut dec = iced_x86::Decoder::with_ip(64, &bytes, 0x1000, 0);
    let mut fmt = iced_x86::NasmFormatter::new();
    let mut text = Vec::new();
    while dec.can_decode() {
        let insn = dec.decode();
        let mut t = String::new();
        iced_x86::Formatter::format(&mut fmt, &insn, &mut t);
        text.push(t);
    }
    let cmp = text
        .iter()
        .find(|t| t.starts_with("cmp") || t.starts_with("test"))
        .unwrap_or_else(|| panic!("no comparison in {text:?}"));
    assert!(
        cmp.contains("rcx"),
        "the comparison must name rcx, the register the argument arrives in; \
         any other register decides the branch on an unrelated value. got {cmp:?}"
    );
}

/// Assemble one instruction and return its disassembly.
pub(super) fn text_of(instrs: &[Instr], alloc: &Alloc, layout: &FrameLayout) -> Vec<String> {
    let mut asm = CodeAssembler::new(64).expect("assembler");
    for (pos, i) in instrs.iter().enumerate() {
        emit_instr(
            &mut asm,
            alloc,
            i,
            pos,
            layout,
            ImageRange::none(),
            Some(instrs),
            &CallRedirects::none(),
        )
        .expect("emit");
    }
    let bytes = asm.assemble(0x1000).expect("assemble");
    let mut dec = iced_x86::Decoder::with_ip(64, &bytes, 0x1000, 0);
    let mut fmt = iced_x86::NasmFormatter::new();
    let mut out = Vec::new();
    while dec.can_decode() {
        let insn = dec.decode();
        let mut t = String::new();
        iced_x86::Formatter::format(&mut fmt, &insn, &mut t);
        out.push(t);
    }
    out
}

/// A return value the allocator homes outside RAX is moved there from RAX.
#[test]
fn a_return_value_homed_outside_rax_is_moved_from_rax() {
    let ret = ValId(0);
    let instrs = vec![Instr::Opaque {
        dst: ret,
        tag: "call_ret",
        width: Width::W64,
        at: Some(Reg::Rax),
    }];
    let alloc = Alloc {
        segments: vec![Segment {
            item: Item::Val(ret),
            start: 0,
            end: 4,
            loc: Loc::Reg(Reg::Rcx),
        }],
        ..Alloc::default()
    };
    let text = text_of(&instrs, &alloc, &FrameLayout::new(0, 0, false));
    assert_eq!(
        text,
        vec!["mov rcx,rax".to_string()],
        "a return value homed in rcx must be copied out of rax, or every read of \
         it reads whatever rcx happened to hold"
    );
}

/// A spilled return value is stored from RAX to its assigned slot.
#[test]
fn a_spilled_return_value_is_stored_to_its_slot() {
    let ret = ValId(0);
    let instrs = vec![Instr::Opaque {
        dst: ret,
        tag: "call_ret",
        width: Width::W64,
        at: Some(Reg::Rax),
    }];
    let alloc = Alloc {
        segments: vec![Segment {
            item: Item::Val(ret),
            start: 0,
            end: 4,
            loc: Loc::Spill(0),
        }],
        slots: 1,
        ..Alloc::default()
    };
    let layout = FrameLayout::new(1, 0, false);
    let text = text_of(&instrs, &alloc, &layout);
    let disp = layout.spill_disp(0);
    assert_eq!(
        text.len(),
        1,
        "exactly one store is needed to define a spilled return value, got {text:?}"
    );
    assert!(
        text[0].starts_with("mov ") && text[0].contains("rax") && text[0].contains("rsp"),
        "a spilled return value must be stored from rax to its slot at rsp+{disp:#x}, \
         or readers load an unwritten slot. got {text:?}"
    );
}

/// An opaque with no architectural home emits nothing.
#[test]
fn an_opaque_with_no_home_emits_no_instruction() {
    let v = ValId(0);
    for tag in ["af_undef", "clobbered", "flag", "xreg"] {
        let instrs = vec![Instr::Opaque {
            dst: v,
            tag,
            width: Width::W64,
            at: None,
        }];
        let alloc = Alloc {
            segments: vec![Segment {
                item: Item::Val(v),
                start: 0,
                end: 4,
                loc: Loc::Reg(Reg::Rcx),
            }],
            ..Alloc::default()
        };
        let text = text_of(&instrs, &alloc, &FrameLayout::new(0, 0, false));
        assert!(
            text.is_empty(),
            "{tag:?} has no definition to emit, got {text:?}"
        );
    }
}

/// A narrow control value is tested at its own width because only that subregister
/// is defined.
#[test]
fn a_narrow_control_value_is_tested_at_its_own_width() {
    let v = ValId(0);
    let mut arena = crate::ir::expr::Arena::new();
    let predicate = arena.constant(1, Width::W8);
    let taken_id = BlockId {
        handler: 0x2000,
        vip: None,
    };
    let not_taken_id = BlockId {
        handler: 0x3000,
        vip: None,
    };
    let b = SchedBlock {
        id: BlockId {
            handler: 0x1000,
            vip: None,
        },
        // The comparison is materialised rather than fused, so it stays in
        // `instrs` and its `sete` writes only the low byte of the host register.
        instrs: vec![Instr::Bin {
            dst: v,
            op: crate::ir::expr::BinOp::Eq,
            a: Operand::Entry(Reg::Rcx),
            b: Operand::Imm(0),
            width: Width::W8,
            operand_width: Width::W64,
        }],
        exits: Vec::new(),
        callee_set: Vec::new(),
        control: Some(Operand::Val(v)),
        terminator: Terminator::Branch {
            predicate,
            taken: taken_id,
            not_taken: not_taken_id,
        },
        fused_cmp: None,
    };
    let alloc = Alloc {
        segments: vec![Segment {
            item: Item::Val(v),
            start: 0,
            end: 2,
            loc: Loc::Reg(Reg::Rbx),
        }],
        copies: Vec::new(),
        slots: 0,
    };
    let layout = FrameLayout::new(-0x30, 0, true);

    let mut asm = CodeAssembler::new(64).expect("assembler");
    let mut label_of = HashMap::new();
    let mut taken = asm.create_label();
    let mut not_taken = asm.create_label();
    label_of.insert(taken_id, taken);
    label_of.insert(not_taken_id, not_taken);
    emit_terminator(
        &mut asm,
        &alloc,
        &b,
        &[],
        &label_of,
        &layout,
        None,
        &CallRedirects::none(),
    )
    .expect("emit");
    asm.set_label(&mut taken).expect("label");
    asm.nop().expect("nop");
    asm.set_label(&mut not_taken).expect("label");
    asm.nop().expect("nop");
    let bytes = asm.assemble(0x1000).expect("assemble");
    let mut dec = iced_x86::Decoder::with_ip(64, &bytes, 0x1000, 0);
    let mut fmt = iced_x86::NasmFormatter::new();
    let mut text = Vec::new();
    while dec.can_decode() {
        let insn = dec.decode();
        let mut t = String::new();
        iced_x86::Formatter::format(&mut fmt, &insn, &mut t);
        text.push(t);
    }
    let test = text
        .iter()
        .find(|t| t.starts_with("test"))
        .unwrap_or_else(|| panic!("no test in {text:?}"));
    assert_eq!(
        test, "test bl,bl",
        "a W8 control value must be tested as a byte; testing the full register \
         reads the 56 stale bits above the boolean. got {text:?}"
    );
}

/// Boxed register results are emitted as one parallel copy.
#[test]
fn a_boxed_instructions_results_move_as_a_parallel_copy() {
    let (a, b, c, d) = (ValId(0), ValId(1), ValId(2), ValId(3));
    // `cpuid` is 0f a2.
    let instrs = vec![
        Instr::Boxed {
            site: 0x2000,
            text: "cpuid".to_string(),
            bytes: vec![0x0f, 0xa2],
            mem: None,
            uses: Vec::new(),
        },
        Instr::Opaque {
            dst: a,
            tag: "cpuid",
            width: Width::W64,
            at: Some(Reg::Rax),
        },
        Instr::Opaque {
            dst: b,
            tag: "cpuid",
            width: Width::W64,
            at: Some(Reg::Rbx),
        },
        Instr::Opaque {
            dst: c,
            tag: "cpuid",
            width: Width::W64,
            at: Some(Reg::Rcx),
        },
        Instr::Opaque {
            dst: d,
            tag: "cpuid",
            width: Width::W64,
            at: Some(Reg::Rdx),
        },
    ];
    let seg = |item, loc| Segment {
        item,
        start: 0,
        end: 8,
        loc,
    };
    let alloc = Alloc {
        segments: vec![
            seg(Item::Val(a), Loc::Reg(Reg::Rax)),
            seg(Item::Val(b), Loc::Reg(Reg::Rcx)),
            seg(Item::Val(c), Loc::Reg(Reg::Rdx)),
            seg(Item::Val(d), Loc::Reg(Reg::Rbx)),
        ],
        copies: Vec::new(),
        slots: 0,
    };
    let text = text_of(&instrs, &alloc, &FrameLayout::new(0, 0, false));

    // Simulate the emitted moves against the four results to check the permutation
    // rather than pinning one particular instruction sequence.
    let mut reg: HashMap<&str, &str> = [("rax", "A"), ("rbx", "B"), ("rcx", "C"), ("rdx", "D")]
        .into_iter()
        .collect();
    for t in text.iter().skip(1) {
        let ops: Vec<&str> = t
            .split_whitespace()
            .skip(1)
            .collect::<Vec<_>>()
            .join("")
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| Box::leak(s.to_string().into_boxed_str()) as &str)
            .collect();
        assert_eq!(ops.len(), 2, "unexpected move {t:?} in {text:?}");
        let (dst, src) = (ops[0], ops[1]);
        if t.starts_with("mov") {
            let v = reg[src];
            reg.insert(dst, v);
        } else if t.starts_with("xchg") {
            let (x, y) = (reg[dst], reg[src]);
            reg.insert(dst, y);
            reg.insert(src, x);
        } else {
            panic!("unexpected instruction {t:?} in {text:?}");
        }
    }
    // Each value must end up in the register the allocator assigned it.
    assert_eq!(reg["rax"], "A", "RAX result lost; got {text:?}");
    assert_eq!(reg["rcx"], "B", "RBX result must land in RCX; got {text:?}");
    assert_eq!(reg["rdx"], "C", "RCX result must land in RDX; got {text:?}");
    assert_eq!(reg["rbx"], "D", "RDX result must land in RBX; got {text:?}");
}

/// A 16-bit load through a register reads 16 bits, not 32.
#[test]
fn a_sixteen_bit_load_through_a_register_reads_sixteen_bits() {
    let (addr, loaded) = (ValId(0), ValId(1));
    let instrs = vec![Instr::Load {
        dst: loaded,
        addr: Operand::Val(addr),
        disp: 2,
        width: Width::W16,
    }];
    let seg = |item, loc| Segment {
        item,
        start: 0,
        end: 4,
        loc,
    };
    let alloc = Alloc {
        segments: vec![
            seg(Item::Val(addr), Loc::Reg(Reg::Rcx)),
            seg(Item::Val(loaded), Loc::Reg(Reg::Rax)),
        ],
        copies: Vec::new(),
        slots: 0,
    };
    let text = text_of(&instrs, &alloc, &FrameLayout::new(0, 0, false));
    let load = text
        .iter()
        .find(|t| t.starts_with("mov") && t.contains('['))
        .unwrap_or_else(|| panic!("no load emitted, got {text:?}"));
    // The destination register spelling is what fixes the access size: `ax` reads
    // two bytes, `eax` reads four.
    assert!(
        load.contains(" ax,") || load.contains("word ptr"),
        "a W16 load must read 16 bits; a 32-bit destination reads two bytes past \
         the field the IR named. got {load:?} in {text:?}"
    );
}

/// An 8-bit shift by a constant emits an 8-bit shift.
///
/// The cost was not a wrong instruction but two whole functions: the `bail!`
/// propagates out of `emit_function` and `write_devirt` drops the function
/// entirely, leaving its trampoline pointing at the original VM.
#[test]
fn an_eight_bit_shift_by_a_constant_is_emitted() {
    for (op, want) in [
        (BinOp::Shr, "shr"),
        (BinOp::Shl, "shl"),
        (BinOp::Sar, "sar"),
    ] {
        let (a, dst) = (ValId(0), ValId(1));
        let instrs = vec![Instr::Bin {
            dst,
            op,
            a: Operand::Val(a),
            b: Operand::Imm(1),
            width: Width::W8,
            operand_width: Width::W8,
        }];
        let seg = |item, loc| Segment {
            item,
            start: 0,
            end: 4,
            loc,
        };
        let alloc = Alloc {
            segments: vec![
                seg(Item::Val(a), Loc::Reg(Reg::Rcx)),
                seg(Item::Val(dst), Loc::Reg(Reg::Rcx)),
            ],
            copies: Vec::new(),
            slots: 0,
        };
        let text = text_of(&instrs, &alloc, &FrameLayout::new(0x20, 0, false));
        let shift = text
            .iter()
            .find(|t| t.starts_with(want))
            .unwrap_or_else(|| panic!("no 8-bit {op:?} emitted, got {text:?}"));
        // An 8-bit shift must operate on an 8-bit register: shifting `cx` or `ecx`
        // brings bits above bit 7 into the result.
        assert!(
            shift.contains("cl,"),
            "an 8-bit {op:?} must shift an 8-bit register, got {shift:?} in {text:?}"
        );
    }
}

/// Two blocks in a row can each get a label even if the first emits nothing.
///
/// A `nop` between them is what separates the labels. Asserting on iced directly
/// keeps the test honest about *why* the fix is needed rather than restating it.
#[test]
fn two_labels_need_an_instruction_between_them() {
    let mut asm = CodeAssembler::new(64).expect("assembler");
    let mut a = asm.create_label();
    let mut b = asm.create_label();
    asm.set_label(&mut a).expect("first label");
    // Without this, the second `set_label` fails.
    asm.nop().expect("separator");
    asm.set_label(&mut b)
        .expect("a second label needs its own instruction to attach to");
    asm.ret().expect("ret");
    assert!(asm.assemble(0x1000).is_ok(), "should assemble");

    // And confirm the failure really is what the fix guards against.
    let mut asm = CodeAssembler::new(64).expect("assembler");
    let mut a = asm.create_label();
    let mut b = asm.create_label();
    asm.set_label(&mut a).expect("first label");
    assert!(
        asm.set_label(&mut b).is_err(),
        "iced must reject two labels on one instruction; if this ever becomes \
         allowed the nop in emit_function is dead weight"
    );
}

/// A boxed instruction's implicit register reads are materialized before it.
/// Fixed inputs such as CPUID's EAX/ECX and string instructions' RDI/RCX/AL cannot
/// be rewritten in the original encoding.
#[test]
fn a_boxed_instructions_implicit_reads_are_materialized() {
    let (leaf, sub) = (ValId(0), ValId(1));
    // `cpuid` reads EAX and ECX without naming either as a rewritable operand.
    let instrs = vec![Instr::Boxed {
        site: 0x2000,
        text: "cpuid".to_string(),
        bytes: vec![0x0f, 0xa2],
        mem: None,
        uses: vec![(Reg::Rax, Operand::Imm(7)), (Reg::Rcx, Operand::Val(sub))],
    }];
    let seg = |item, loc| Segment {
        item,
        start: 0,
        end: 4,
        loc,
    };
    let alloc = Alloc {
        segments: vec![
            seg(Item::Val(leaf), Loc::Reg(Reg::Rbx)),
            seg(Item::Val(sub), Loc::Reg(Reg::Rdx)),
        ],
        copies: Vec::new(),
        slots: 0,
    };
    let text = text_of(&instrs, &alloc, &FrameLayout::new(0x20, 0, false));
    let cpuid = text
        .iter()
        .position(|t| t.starts_with("cpuid"))
        .unwrap_or_else(|| panic!("no cpuid emitted, got {text:?}"));
    assert!(
        cpuid > 0,
        "the inputs must be set up *before* the instruction, got {text:?}"
    );
    let setup = &text[..cpuid];
    // The leaf is an immediate; it has to reach EAX somehow.
    assert!(
        setup.iter().any(|t| t.contains("eax") || t.contains("rax")),
        "cpuid's leaf must be materialized into (E)AX, got {setup:?}"
    );
    // The subleaf lives in RDX and has to be moved to RCX.
    assert!(
        setup.iter().any(|t| t.contains("rcx,rdx")),
        "cpuid's subleaf must be moved from where it lives into RCX, got {setup:?}"
    );
}

/// A variable shift must take its count from the register the count actually lives
/// in, not from CL unconditionally.
#[test]
fn a_variable_shift_takes_its_count_from_the_counts_own_register() {
    for (op, want) in [
        (BinOp::Shl, "shl"),
        (BinOp::Shr, "shr"),
        (BinOp::Sar, "sar"),
        (BinOp::Rol, "rol"),
        (BinOp::Ror, "ror"),
    ] {
        let (a, b, dst) = (ValId(0), ValId(1), ValId(2));
        let instrs = vec![Instr::Bin {
            dst,
            op,
            a: Operand::Val(a),
            b: Operand::Val(b),
            width: Width::W64,
            operand_width: Width::W64,
        }];
        let seg = |item, loc| Segment {
            item,
            start: 0,
            end: 4,
            loc,
        };
        let alloc = Alloc {
            // The count is in R8. RCX holds nothing.
            segments: vec![
                seg(Item::Val(a), Loc::Reg(Reg::Rax)),
                seg(Item::Val(b), Loc::Reg(Reg::R8)),
                seg(Item::Val(dst), Loc::Reg(Reg::Rax)),
            ],
            copies: Vec::new(),
            slots: 0,
        };
        let text = text_of(&instrs, &alloc, &FrameLayout::new(0x20, 0, false));
        // The count has to reach RCX, and R8 has to be given back afterwards, so the
        // sequence is a swap, the shift, and the swap undone.
        assert_eq!(
            text.iter().filter(|t| t.starts_with("xchg")).count(),
            2,
            "the count must be swapped into RCX and swapped back, got {text:?}"
        );
        assert!(
            text.iter().any(|t| t == &format!("{want} rax,cl")),
            "expected `{want} rax,cl` between the swaps, got {text:?}"
        );
    }
}

/// A rotate by a constant uses the immediate form.
#[test]
fn a_rotate_by_a_constant_uses_the_immediate_form() {
    for (op, want) in [(BinOp::Rol, "rol rax,3"), (BinOp::Ror, "ror rax,3")] {
        let (a, dst) = (ValId(0), ValId(1));
        let instrs = vec![Instr::Bin {
            dst,
            op,
            a: Operand::Val(a),
            b: Operand::Imm(3),
            width: Width::W64,
            operand_width: Width::W64,
        }];
        let seg = |item, loc| Segment {
            item,
            start: 0,
            end: 4,
            loc,
        };
        let alloc = Alloc {
            segments: vec![
                seg(Item::Val(a), Loc::Reg(Reg::Rax)),
                seg(Item::Val(dst), Loc::Reg(Reg::Rax)),
            ],
            copies: Vec::new(),
            slots: 0,
        };
        let text = text_of(&instrs, &alloc, &FrameLayout::new(0x20, 0, false));
        assert_eq!(text, vec![want.to_string()], "got {text:?}");
    }
}

/// `b` sharing the destination register must not be clobbered by loading `a`. The allocator's handoff rule lets a value that dies at an instruction share a register with the value that instruction defines.
#[test]
fn a_second_operand_sharing_the_destination_register_is_moved_first() {
    let (a, b, dst) = (ValId(0), ValId(1), ValId(2));
    let instrs = vec![Instr::Bin {
        dst,
        op: BinOp::Sub,
        a: Operand::Val(a),
        b: Operand::Val(b),
        width: Width::W64,
        operand_width: Width::W64,
    }];
    let alloc = Alloc {
        segments: vec![
            Segment {
                item: Item::Val(a),
                start: 0,
                end: 1,
                loc: Loc::Reg(Reg::R8),
            },
            // `b` dies here and `dst` is born here, both in RAX.
            Segment {
                item: Item::Val(b),
                start: 0,
                end: 1,
                loc: Loc::Reg(Reg::Rax),
            },
            Segment {
                item: Item::Val(dst),
                start: 0,
                end: 4,
                loc: Loc::Reg(Reg::Rax),
            },
        ],
        copies: Vec::new(),
        slots: 0,
    };
    let text = text_of(&instrs, &alloc, &FrameLayout::new(0x20, 0, false));
    // Whatever register `b` is moved to, the subtraction must not name RAX twice.
    assert!(
        !text.iter().any(|t| t == "sub rax,rax"),
        "`a - b` was lowered as `a - a`, got {text:?}"
    );
    let sub = text
        .iter()
        .find(|t| t.starts_with("sub "))
        .unwrap_or_else(|| panic!("no subtraction emitted, got {text:?}"));
    let (_, ops) = sub.split_once(' ').expect("a sub has operands");
    let (l, r) = ops.split_once(',').expect("a sub has two operands");
    assert_ne!(
        l, r,
        "the subtraction must name two different registers, got {text:?}"
    );
    // And `b` must be read from where it was moved, so that register has to be
    // written before the subtraction reads it.
    assert!(
        text.iter()
            .any(|t| t.starts_with("mov") && t.ends_with(",rax")),
        "`b` must be moved out of the destination register first, got {text:?}"
    );
}

/// A high-half multiply is emitted rather than dropping the whole function.
///
/// RAX and RDX are written by `mul` whatever the allocation says, so both are
/// preserved around the sequence.
#[test]
fn a_high_half_multiply_saves_and_restores_the_fixed_registers() {
    let (a, b, dst) = (ValId(0), ValId(1), ValId(2));
    let instrs = vec![Instr::Bin {
        dst,
        op: BinOp::MulHiU,
        a: Operand::Val(a),
        b: Operand::Val(b),
        width: Width::W64,
        operand_width: Width::W64,
    }];
    let seg = |item, loc| Segment {
        item,
        start: 0,
        end: 4,
        loc,
    };
    let alloc = Alloc {
        // The destination is neither RAX nor RDX, so both must be given back.
        segments: vec![
            seg(Item::Val(a), Loc::Reg(Reg::R9)),
            seg(Item::Val(b), Loc::Reg(Reg::R10)),
            seg(Item::Val(dst), Loc::Reg(Reg::R11)),
        ],
        copies: Vec::new(),
        slots: 0,
    };
    let layout = FrameLayout::new(0x20, 0, false);
    let text = text_of(&instrs, &alloc, &layout);
    assert!(
        text.iter().any(|t| t.starts_with("mul ")),
        "a MulHiU must emit a `mul`, got {text:?}"
    );
    // The high half is in RDX, and it has to be taken out before RDX is restored.
    let mul = text
        .iter()
        .position(|t| t.starts_with("mul "))
        .expect("a mul");
    let take = text
        .iter()
        .position(|t| t == "mov r11,rdx")
        .unwrap_or_else(|| panic!("the high half must be moved to the destination, got {text:?}"));
    assert!(
        take > mul,
        "the result must be read after the multiply, got {text:?}"
    );
    // Both fixed registers are saved before and restored after.
    for r in ["rax", "rdx"] {
        assert!(
            text.iter()
                .any(|t| t.starts_with(&format!("mov [rsp+")) && t.ends_with(&format!("],{r}"))),
            "{r} must be preserved before the multiply, got {text:?}"
        );
        let restore = text
            .iter()
            .position(|t| t.starts_with(&format!("mov {r},[rsp+")))
            .unwrap_or_else(|| panic!("{r} must be restored after the multiply, got {text:?}"));
        assert!(
            restore > take,
            "{r} must be restored after the result is read, got {text:?}"
        );
    }
}

/// An 8-bit multiply rides the 32-bit form.
#[test]
fn an_eight_bit_multiply_uses_the_thirty_two_bit_form() {
    let (a, b, dst) = (ValId(0), ValId(1), ValId(2));
    let instrs = vec![Instr::Bin {
        dst,
        op: BinOp::Mul,
        a: Operand::Val(a),
        b: Operand::Val(b),
        width: Width::W8,
        operand_width: Width::W8,
    }];
    let seg = |item, loc| Segment {
        item,
        start: 0,
        end: 4,
        loc,
    };
    let alloc = Alloc {
        segments: vec![
            seg(Item::Val(a), Loc::Reg(Reg::Rax)),
            seg(Item::Val(b), Loc::Reg(Reg::R8)),
            seg(Item::Val(dst), Loc::Reg(Reg::Rax)),
        ],
        copies: Vec::new(),
        slots: 0,
    };
    let text = text_of(&instrs, &alloc, &FrameLayout::new(0x20, 0, false));
    assert!(
        text.iter().any(|t| t == "imul eax,r8d"),
        "an 8-bit Mul should ride the 32-bit imul, got {text:?}"
    );
}

/// The register holding the multiplicand must not be the one holding `a`.
#[test]
fn a_high_half_multiply_does_not_place_its_operand_over_a() {
    let (a, b, dst) = (ValId(0), ValId(1), ValId(2));
    let instrs = vec![Instr::Bin {
        dst,
        op: BinOp::MulHiU,
        a: Operand::Val(a),
        b: Operand::Imm(0x51EB851F),
        width: Width::W32,
        operand_width: Width::W32,
    }];
    let seg = |item, loc| Segment {
        item,
        start: 0,
        end: 4,
        loc,
    };
    let alloc = Alloc {
        // `a` sits in RCX, which is the first register the holder search would
        // otherwise reach.
        segments: vec![
            seg(Item::Val(a), Loc::Reg(Reg::Rcx)),
            seg(Item::Val(b), Loc::Reg(Reg::Rcx)),
            seg(Item::Val(dst), Loc::Reg(Reg::Rax)),
        ],
        copies: Vec::new(),
        slots: 0,
    };
    let text = text_of(&instrs, &alloc, &FrameLayout::new(0x20, 0, false));
    // The multiplicand must not be RCX, since RCX still has to carry `a` into RAX.
    assert!(
        !text.iter().any(|t| t == "mul ecx"),
        "the multiplicand was placed over `a`, got {text:?}"
    );
    let load_a = text
        .iter()
        .position(|t| t == "mov eax,ecx")
        .unwrap_or_else(|| panic!("`a` must be read from RCX, got {text:?}"));
    let mul = text
        .iter()
        .position(|t| t.starts_with("mul "))
        .unwrap_or_else(|| panic!("no multiply emitted, got {text:?}"));
    assert!(
        load_a < mul,
        "`a` must reach RAX before the multiply, got {text:?}"
    );
}

/// A value shifted by itself must not be swapped out from under the shift. `x << x` is legal IR and the allocator's handoff rule lets the value and the count share a register.
#[test]
fn a_value_shifted_by_itself_stays_in_its_register() {
    let (a, dst) = (ValId(0), ValId(1));
    let instrs = vec![Instr::Bin {
        dst,
        op: BinOp::Shl,
        a: Operand::Val(a),
        b: Operand::Val(a),
        width: Width::W64,
        operand_width: Width::W64,
    }];
    let seg = |item, loc| Segment {
        item,
        start: 0,
        end: 4,
        loc,
    };
    let alloc = Alloc {
        segments: vec![
            seg(Item::Val(a), Loc::Reg(Reg::R8)),
            seg(Item::Val(dst), Loc::Reg(Reg::R8)),
        ],
        copies: Vec::new(),
        slots: 0,
    };
    let text = text_of(&instrs, &alloc, &FrameLayout::new(0x20, 0, false));
    assert!(
        !text.iter().any(|t| t.starts_with("xchg")),
        "a self-shift must not swap the value out of its register, got {text:?}"
    );
    assert!(
        text.iter().any(|t| t == "shl r8,cl"),
        "the shift must operate on the value in place, got {text:?}"
    );
    // The count is copied in, and RCX; which the allocation does not describe as
    // free; is given back.
    let copy = text
        .iter()
        .position(|t| t == "mov rcx,r8")
        .unwrap_or_else(|| panic!("the count must be copied to RCX, got {text:?}"));
    let shift = text
        .iter()
        .position(|t| t == "shl r8,cl")
        .expect("the shift");
    assert!(
        copy < shift,
        "the count must reach CL before the shift, got {text:?}"
    );
    assert!(
        text.iter().any(|t| t.starts_with("mov rcx,[rsp+")),
        "RCX held a live value and must be restored, got {text:?}"
    );
}

/// Every scratch slot lies inside the frame and below the guest locals. The RDX:RAX opcodes and the self-shift path preserve registers in slots 1..3, which grew `SCRATCH_BYTES` from one qword to four.
#[test]
fn the_scratch_slots_fit_inside_the_frame() {
    for locals in [0i64, -8, -0x268, -0x1000] {
        for spills in [0u32, 1, 7, 40] {
            for saves in [vec![], vec![Reg::Rbx], vec![Reg::Rbx, Reg::Rbp, Reg::R12]] {
                for calls in [false, true] {
                    let l = FrameLayout::full(locals, spills, calls, saves.clone(), 0);
                    let frame = l.frame_size as i64;
                    // The lowest guest local sits at `frame_size + locals`.
                    let locals_lo = frame + locals;
                    for i in 0..4u32 {
                        let d = l.scratch_slot_disp(i);
                        assert!(
                            d >= 0 && d + 8 <= frame,
                            "scratch slot {i} at {d:#x} escapes a {frame:#x}-byte frame                                  (locals={locals:#x} spills={spills} saves={} calls={calls})",
                            saves.len()
                        );
                        assert!(
                            d + 8 <= locals_lo,
                            "scratch slot {i} at {d:#x} overlaps the guest locals at                                  {locals_lo:#x} (spills={spills} saves={} calls={calls})",
                            saves.len()
                        );
                    }
                    // And the slots are distinct, so three saves cannot alias.
                    for i in 0..4u32 {
                        for j in 0..i {
                            assert_ne!(
                                l.scratch_slot_disp(i),
                                l.scratch_slot_disp(j),
                                "scratch slots {i} and {j} alias"
                            );
                        }
                    }
                    // Nor may a scratch slot alias a spill slot or a save area slot.
                    for i in 0..4u32 {
                        let d = l.scratch_slot_disp(i);
                        for s in 0..spills {
                            assert_ne!(d, l.spill_disp(s), "scratch {i} aliases spill {s}");
                        }
                        for r in &saves {
                            assert_ne!(
                                Some(d),
                                l.save_disp(*r),
                                "scratch {i} aliases the save slot for {r:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}
