use super::*;
use iced_x86::{Decoder, DecoderOptions};

/// One frame-establishing instruction in the prologue. Only these two shapes exist: `emit_function` reserves the frame with a single `sub rsp` and then stores each callee-saved register with a `mov`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrologueStep {
    /// `sub rsp, bytes`.
    Alloc { bytes: u32 },
    /// `mov [rsp + disp], reg`, with `disp` measured from the post-`sub` RSP.
    Save { reg: Reg, disp: u32 },
}

/// Byte offset of the end of each prologue instruction.
///
/// The unwind record's `CodeOffset` counts the byte *following* the instruction, so
/// these are cumulative lengths rather than starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrologueInstr {
    pub end: u8,
    pub step: PrologueStep,
}

/// Read the prologue's instruction lengths back out of the assembled bytes.
pub(crate) fn measure_prologue(bytes: &[u8], layout: &FrameLayout) -> Vec<PrologueInstr> {
    let mut steps: Vec<PrologueStep> = Vec::new();
    if layout.frame_size > 0 {
        steps.push(PrologueStep::Alloc {
            bytes: layout.frame_size,
        });
    }
    for &r in layout.saves() {
        let disp = layout
            .save_disp(r)
            .expect("save slot for a register in saves()");
        steps.push(PrologueStep::Save {
            reg: r,
            disp: disp as u32,
        });
    }

    let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
    let mut out = Vec::with_capacity(steps.len());
    let mut end = 0usize;
    for step in steps {
        let instr = decoder.decode();
        end += instr.len();
        // A prologue longer than 255 bytes cannot be described at all, and the format gives no way to say so.
        if end > u8::MAX as usize {
            break;
        }
        out.push(PrologueInstr {
            end: end as u8,
            step,
        });
    }
    out
}

/// The register number an `UNWIND_CODE`'s `OpInfo` field uses. This is the architectural encoding, which is *not* the order `Reg` happens to be declared in, so it is spelled out.
pub(crate) fn unwind_reg_code(r: Reg) -> u8 {
    match r {
        Reg::Rax => 0,
        Reg::Rcx => 1,
        Reg::Rdx => 2,
        Reg::Rbx => 3,
        Reg::Rsp => 4,
        Reg::Rbp => 5,
        Reg::Rsi => 6,
        Reg::Rdi => 7,
        Reg::R8 => 8,
        Reg::R9 => 9,
        Reg::R10 => 10,
        Reg::R11 => 11,
        Reg::R12 => 12,
        Reg::R13 => 13,
        Reg::R14 => 14,
        Reg::R15 => 15,
    }
}

/// Build the Win64 `UNWIND_INFO` blob describing an emitted function's frame. Without one, `RtlLookupFunctionEntry` finds no entry for the address and the unwinder treats the function as a leaf: RSP unchanged, no registers saved.
pub fn unwind_info(prologue: &[PrologueInstr]) -> Vec<u8> {
    // `SizeOfProlog` is where the last frame-establishing instruction ends.
    let prolog_size = prologue.last().map_or(0, |p| p.end);

    // Unwind codes run in *decreasing* `CodeOffset`, the reverse of program order, because the unwinder walks them to undo a partially-executed prologue.
    let mut codes: Vec<u8> = Vec::new();
    for pi in prologue.iter().rev() {
        match pi.step {
            PrologueStep::Save { reg, disp } => {
                // The saves are `mov [rsp+disp], reg`, so UWOP_SAVE_NONVOL, whose offset is scaled by 8.
                debug_assert_eq!(disp % 8, 0, "a save slot must be qword-aligned");
                let scaled = disp / 8;
                if scaled <= u16::MAX as u32 {
                    codes.push(pi.end);
                    codes.push((unwind_reg_code(reg) << 4) | UWOP_SAVE_NONVOL);
                    codes.extend_from_slice(&(scaled as u16).to_le_bytes());
                } else {
                    codes.push(pi.end);
                    codes.push((unwind_reg_code(reg) << 4) | UWOP_SAVE_NONVOL_FAR);
                    codes.extend_from_slice(&disp.to_le_bytes());
                }
            }
            PrologueStep::Alloc { bytes } => {
                // UWOP_ALLOC_SMALL covers 8..=128 in one slot, encoding the size as `bytes/8 - 1`. Anything larger needs UWOP_ALLOC_LARGE: OpInfo 0 for a size scaled by 8 into one extra slot, OpInfo 1 for the raw byte count in two.
                debug_assert_eq!(bytes % 8, 0, "the frame size must be qword-aligned");
                if (8..=128).contains(&bytes) {
                    codes.push(pi.end);
                    codes.push((((bytes / 8 - 1) as u8) << 4) | UWOP_ALLOC_SMALL);
                } else if bytes / 8 <= u16::MAX as u32 {
                    codes.push(pi.end);
                    codes.push(UWOP_ALLOC_LARGE);
                    codes.extend_from_slice(&((bytes / 8) as u16).to_le_bytes());
                } else {
                    codes.push(pi.end);
                    codes.push((1 << 4) | UWOP_ALLOC_LARGE);
                    codes.extend_from_slice(&bytes.to_le_bytes());
                }
            }
        }
    }

    debug_assert_eq!(codes.len() % 2, 0, "unwind codes are two-byte slots");
    let mut info = Vec::with_capacity(4 + codes.len() + 2);
    // Version 1, no flags: no exception or termination handler, and no chaining. The
    // devirtualized body has no `try` regions of its own; any handler the original
    // function had lives in the caller's scope table, which we do not touch.
    info.push(UNWIND_VERSION_1);
    info.push(prolog_size);
    info.push((codes.len() / 2) as u8);
    // No frame register: everything is addressed off RSP, which never moves after the
    // prologue. That is what makes the whole record this simple.
    info.push(0);
    info.extend_from_slice(&codes);
    // The structure is a u32 array, so an odd number of code slots leaves a trailing
    // half-slot that must be padded.
    while info.len() % 4 != 0 {
        info.push(0);
    }
    info
}

pub(crate) const UNWIND_VERSION_1: u8 = 1;
pub(crate) const UWOP_ALLOC_LARGE: u8 = 1;
pub(crate) const UWOP_ALLOC_SMALL: u8 = 2;
pub(crate) const UWOP_SAVE_NONVOL: u8 = 4;
pub(crate) const UWOP_SAVE_NONVOL_FAR: u8 = 5;
