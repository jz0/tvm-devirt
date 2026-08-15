use crate::binary::disasm::decode_at;
use crate::binary::pe::PeFile;

/// Whether a candidate `E9` at `va` is contradicted by the instruction stream.
///
/// A real trampoline is the *first* instruction of a function, so it always sits
/// on an instruction boundary. A chance `E9` byte inside some other instruction's
/// encoding does not. That difference is the only thing separating the two, and
/// `.pdata` gives us a trustworthy place to start decoding from to tell them
/// apart.
///
/// Returns `true` only with positive evidence of a mid-instruction hit: the sweep
/// from the enclosing function's start stepped *over* `va` without landing on it.
/// Three cases, and only the second is a rejection:
///
/// * sweep lands on `va`; real boundary, keep.
/// * sweep steps over `va`; inside another instruction, reject.
/// * no enclosing record, or the sweep hits undecodable bytes before reaching
///   `va`; no evidence either way, keep.
fn is_mid_instruction(pe: &PeFile, table: &[(u64, u64)], va: u64) -> bool {
    // Enclosing record: the last one that begins at or before `va`. The table is
    // sorted, and entries do not overlap.
    let idx = match table.binary_search_by_key(&va, |&(begin, _)| begin) {
        Ok(i) => i,
        Err(0) => return false,
        Err(i) => i - 1,
    };
    let (begin, end) = table[idx];
    if va < begin || va >= end {
        return false; // not covered by .pdata
    }

    let mut ip = begin;
    while ip < va {
        match decode_at(pe, ip) {
            Some(i) => ip = i.next_ip(),
            // Undecodable bytes before `va`: the sweep proves nothing past here.
            None => return false,
        }
    }
    // Overshot `va`, so some instruction's encoding contains it.
    ip != va
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VmEntry {
    /// VA of the `E9` trampoline (or the all-`int3` body start for a swallowed
    /// function) in the original code section.
    pub trampoline_va: u64,
    /// VA inside the VM section that the trampoline jumps to.
    pub vm_entry_va: u64,
    /// Number of `int3` bytes directly following the `E9 rel32`.
    /// Zero for swallowed functions that have no `E9` of their own.
    pub int3_padding: usize,
}

/// Scan every executable section other than the VM section for entry
/// trampolines pointing into the VM section.
pub fn find_vm_entries(pe: &PeFile, vm_section: &str) -> Vec<VmEntry> {
    let Some(vm) = pe.section_by_name(vm_section) else {
        return Vec::new();
    };
    let vm_lo = pe.rva_to_va(vm.virtual_address);
    let vm_hi = vm_lo + vm.virtual_size.max(vm.raw_size) as u64;

    let table = pe.function_table();

    let mut out = Vec::new();
    for sec in &pe.sections {
        if sec.name == vm_section || !sec.is_executable() {
            continue;
        }
        let bytes = pe.section_bytes(sec);
        let base = pe.rva_to_va(sec.virtual_address);

        let mut i = 0usize;
        while i + 5 <= bytes.len() {
            if bytes[i] != 0xE9 {
                i += 1;
                continue;
            }
            let rel = i32::from_le_bytes(bytes[i + 1..i + 5].try_into().unwrap());
            let va = base + i as u64;
            let target = va.wrapping_add(5).wrapping_add(rel as i64 as u64);
            if !(vm_lo..vm_hi).contains(&target) {
                i += 1;
                continue;
            }
            // Reject chance `E9` bytes that live inside another instruction.
            if is_mid_instruction(pe, &table, va) {
                i += 1;
                continue;
            }
            let padding = if i + 5 < bytes.len() {
                bytes[i + 5..].iter().take_while(|&&b| b == 0xCC).count()
            } else {
                0
            };
            out.push(VmEntry {
                trampoline_va: va,
                vm_entry_va: target,
                int3_padding: padding,
            });
            // Skip past the trampoline and its padding: nothing else lives there.
            i += 5 + padding.max(1);
        }
    }

    // Second pass: pdata-based discovery for "swallowed" functions.
    let tramp_set: Vec<(u64, u64, u64)> = out
        .iter()
        .map(|e| {
            (
                e.trampoline_va,
                e.trampoline_va + 5 + e.int3_padding as u64,
                e.vm_entry_va,
            )
        })
        .collect();
    let tramp_vas: std::collections::HashSet<u64> = out.iter().map(|e| e.trampoline_va).collect();

    for &(fn_begin, fn_end) in &table {
        // Already discovered as a trampoline.
        if tramp_vas.contains(&fn_begin) {
            continue;
        }
        // Skip functions in the VM section itself.
        if fn_begin >= vm_lo && fn_begin < vm_hi {
            continue;
        }
        // Body must be entirely int3 to qualify as swallowed.
        let len = fn_end.saturating_sub(fn_begin) as usize;
        if len == 0 {
            continue;
        }
        let Some(body) = pe.read_va(fn_begin, len) else {
            continue;
        };
        if body.len() != len || body.iter().any(|&b| b != 0xCC) {
            continue;
        }
        // Must be contained within one trampoline's padding region.
        let Some(&(_, _, vm_entry_va)) = tramp_set
            .iter()
            .find(|&&(ts, te, _)| ts <= fn_begin && fn_end <= te)
        else {
            continue;
        };
        out.push(VmEntry {
            trampoline_va: fn_begin,
            vm_entry_va,
            int3_padding: 0, // no E9 of its own
        });
    }

    out.sort_by_key(|e| e.trampoline_va);
    out.dedup_by_key(|e| e.trampoline_va);
    out
}
