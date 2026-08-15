//! Thin helpers over iced-x86.

use crate::binary::pe::PeFile;
use iced_x86::{Decoder, DecoderOptions, Instruction, OpKind};

/// Decode a single instruction at `va`.
pub fn decode_at(pe: &PeFile, va: u64) -> Option<Instruction> {
    let bytes = pe.read_va(va, 16)?;
    if bytes.is_empty() {
        return None;
    }
    let mut dec = Decoder::with_ip(64, bytes, va, DecoderOptions::NONE);
    let inst = dec.decode();
    inst.is_invalid().then_some(()).map_or(Some(inst), |_| None)
}

/// Decode instructions starting from `va`.
pub fn decode_run(pe: &PeFile, va: u64, max: usize) -> Vec<Instruction> {
    let mut out = Vec::new();
    let mut ip = va;
    while out.len() < max {
        match decode_at(pe, ip) {
            Some(i) => {
                ip = i.next_ip();
                let stop = i.is_invalid();
                out.push(i);
                if stop {
                    break;
                }
            }
            None => break,
        }
    }
    out
}

/// Branch target of a *direct* branch only.
///
/// iced reports `FlowControl::Call` for `SYSCALL`, `VMCALL`, `VMLAUNCH` etc.,
/// where `near_branch64()` silently returns 0. Keying off the operand kind
/// instead of flow control avoids treating those as a branch to address zero.
pub fn direct_target(inst: &Instruction) -> Option<u64> {
    matches!(
        inst.op0_kind(),
        OpKind::NearBranch16 | OpKind::NearBranch32 | OpKind::NearBranch64
    )
    .then(|| inst.near_branch64())
}
