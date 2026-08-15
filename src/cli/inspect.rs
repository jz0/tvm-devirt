//! Read-only inspection of the target: PE layout, discovered entries, raw
//! disassembly, the VM context image, and a single-path emulator trace.

use crate::cli::fmt::{describe_stop, fmt_ref, fmt_ref_depth, resolve_start};
use crate::{binary::disasm, binary::pe, ir::expr, ir::lift, vm::discover};
use anyhow::{Context, Result};
use iced_x86::{Formatter, IntelFormatter};
use std::path::PathBuf;

pub fn cmd_vmctx(
    path: &PathBuf,
    va: u64,
    stack_base: u64,
    steps: usize,
    window: u64,
) -> Result<()> {
    let pe = pe::PeFile::load(path)?;
    let start = resolve_start(&pe, va);
    let mut emu = lift::Emulator::new(&pe, stack_base);
    let stop = emu.run(start, steps);
    println!("stopped: {}", describe_stop(&emu, &stop));

    let Some(rbp) = emu
        .state
        .regs
        .get(&expr::Reg::Rbp)
        .and_then(|r| emu.arena.as_const(*r))
    else {
        println!("RBP is not concrete; cannot locate the context");
        return Ok(());
    };
    println!(
        "rbp = {rbp:#x}, stack_base = {stack_base:#x}
"
    );
    match emu.locate_guest_context() {
        Some(base) => {
            println!(
                "guest register image at {base:#x} (rbp+{:#x})",
                base.wrapping_sub(rbp)
            );
            if let Some(regs) = emu.guest_registers() {
                for (r, v) in &regs {
                    println!("    {r:?} = {}", expr::render(&emu.arena, *v, 3));
                }
            }
        }
        None => println!("no guest register image: mutation-only function, not virtualized"),
    }
    println!();
    if let Some(base) = emu.locate_guest_context() {
        println!("raw slots around the image base (no layout assumed):");
        for i in -4i64..24 {
            let a = base.wrapping_add((i * 8) as u64);
            let shown = match emu.raw_slot(a) {
                Some(v) => expr::render(&emu.arena, v, 3),
                None => "<unset>".to_string(),
            };
            println!("    base{:+#06x}  [{a:#x}] = {shown}", i * 8);
        }
        println!();
    }

    // Walk qword slots across the window and render each.
    let lo = rbp.saturating_sub(0x20);
    let hi = rbp.saturating_add(window);
    let mut addr = lo & !7;
    while addr < hi {
        let v = emu
            .state
            .load_concrete(&mut emu.arena, addr, expr::Width::W64, |a| {
                pe.read_va(a, 1).map(|b| b[0])
            });
        if let Some(v) = v {
            let rendered = expr::render(&emu.arena, v, 3);
            // Suppress slots that were never written and read back as image
            // bytes or zero: they carry no information about the context.
            if rendered != "0x0" {
                println!(
                    "  [rbp+{:#05x}] {addr:#x}  {rendered}",
                    addr.wrapping_sub(rbp)
                );
            }
        }
        addr += 8;
    }
    Ok(())
}

pub fn cmd_trace(
    path: &PathBuf,
    va: u64,
    steps: usize,
    stack_base: u64,
    show_events: bool,
    show_dispatches: bool,
    tail: usize,
    break_at: Option<u64>,
) -> Result<()> {
    let pe = pe::PeFile::load(path)?;

    // Accept an entry trampoline and follow it into the VM section.
    let mut start = va;
    if let Some(inst) = disasm::decode_at(&pe, va) {
        if inst.mnemonic() == iced_x86::Mnemonic::Jmp {
            if let Some(t) = disasm::direct_target(&inst) {
                println!("trampoline {va:#x} -> {t:#x}");
                start = t;
            }
        }
    }

    let mut emu = lift::Emulator::new(&pe, stack_base);
    let t0 = std::time::Instant::now();

    // When a breakpoint is requested, single-step so the state can be dumped
    // exactly at that instruction.
    let stop = match break_at {
        None => emu.run(start, steps),
        Some(target) => {
            let mut ip = start;
            let mut result = None;
            for _ in 0..steps {
                if ip == target {
                    println!("--- state at {target:#x} ---");
                    for r in expr::GPRS {
                        let v = emu.state.reg(r);
                        println!("  {:<4} {}", r.name(), fmt_ref_depth(&emu.arena, v, 6));
                    }
                    println!("---");
                    result = Some(lift::Stop::Budget { site: ip });
                    break;
                }
                match emu.step(ip) {
                    lift::Step::Next(n) => ip = n,
                    lift::Step::Stopped(s) => {
                        result = Some(s);
                        break;
                    }
                }
            }
            result.unwrap_or(lift::Stop::Budget { site: ip })
        }
    };
    let elapsed = t0.elapsed();

    println!(
        "executed {} instructions in {:.1?} ({} DAG nodes)",
        emu.steps,
        elapsed,
        emu.arena.len()
    );
    println!("stopped: {}", describe_stop(&emu, &stop));

    // For a stuck dispatch computation, the useful diagnostic is which symbolic
    // leaves it still depends on.
    if let lift::Stop::SymbolicBranch { dest, .. } = &stop {
        let leaves = expr::leaves(&emu.arena, *dest);
        println!("branch depends on {} symbolic leaves:", leaves.len());
        for l in leaves.iter().take(12) {
            println!("    {}", fmt_ref_depth(&emu.arena, *l, 12));
        }
        if leaves.len() > 12 {
            println!("    ... {} more", leaves.len() - 12);
        }
    }

    let stores = emu
        .events
        .iter()
        .filter(|e| matches!(e, lift::Event::Store { .. }))
        .count();
    let loads = emu
        .events
        .iter()
        .filter(|e| matches!(e, lift::Event::Load { .. }))
        .count();
    let calls = emu
        .events
        .iter()
        .filter(|e| matches!(e, lift::Event::Call { .. }))
        .count();
    println!("events: {stores} stores, {loads} loads, {calls} calls");
    println!("resolved {} VM handler transitions", emu.dispatches.len());

    if !emu.unmapped_loads.is_empty() {
        println!(
            "{} loads from concrete but unmapped addresses (first few):",
            emu.unmapped_loads.len()
        );
        for (site, addr, w) in emu.unmapped_loads.iter().take(8) {
            let delta = addr.wrapping_sub(pe.image_base);
            println!(
                "    @{site:#x} load{} [{addr:#x}]  (image_base + {delta:#x})",
                w.bits()
            );
        }
    }

    if show_dispatches {
        for (i, (site, dest)) in emu.dispatches.iter().enumerate() {
            println!("  [{i:4}] {site:#x} -> {dest:#x}");
        }
    }

    if show_events {
        for e in &emu.events {
            match e {
                lift::Event::Store {
                    addr, width, site, ..
                } => println!(
                    "  store{:<2} @{:#x} addr={}",
                    width.bits(),
                    site,
                    fmt_ref(&emu.arena, *addr)
                ),
                lift::Event::Load {
                    addr, width, site, ..
                } => println!(
                    "  load{:<3} @{:#x} addr={}",
                    width.bits(),
                    site,
                    fmt_ref(&emu.arena, *addr)
                ),
                lift::Event::Call {
                    target, site, rsp, ..
                } => println!(
                    "  call    @{:#x} target={} rsp={}",
                    site,
                    target
                        .map(|t| fmt_ref(&emu.arena, t))
                        .unwrap_or_else(|| "?".into()),
                    rsp.map(|r| format!("{r:#x}"))
                        .unwrap_or_else(|| "symbolic".into())
                ),
                lift::Event::Boxed { site, text, .. } => println!("  boxed   @{site:#x} {text}"),
            }
        }
    }

    if !emu.symbolic_loads.is_empty() {
        println!(
            "{} loads with unresolved addresses; earliest sites:",
            emu.symbolic_loads.len()
        );
        for (site, w) in emu.symbolic_loads.iter().take(8) {
            let mut fmt = IntelFormatter::new();
            let mut s = String::new();
            if let Some(i) = disasm::decode_at(&pe, *site) {
                fmt.format(&i, &mut s);
            }
            println!("    @{site:#x} load{}  {s}", w.bits());
        }
    }

    if tail > 0 {
        let shown = emu.trace_tail(tail);
        println!("last {} instructions:", shown.len());
        let mut fmt = IntelFormatter::new();
        let mut s = String::new();
        for &ip in &shown {
            if let Some(i) = disasm::decode_at(&pe, ip) {
                s.clear();
                fmt.format(&i, &mut s);
                println!("  {ip:>12x}  {s}");
            }
        }
    }

    Ok(())
}

pub fn cmd_info(path: &PathBuf, vm_section: &str) -> Result<()> {
    let pe = pe::PeFile::load(path)?;
    println!("image base   {:#x}", pe.image_base);
    println!("entry point  {:#x}", pe.rva_to_va(pe.entry_point_rva));
    println!("size of image {:#x}", pe.size_of_image);
    println!(
        "\n{:<10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "name", "va", "vsize", "raw", "rsize", "chars"
    );
    for s in &pe.sections {
        println!(
            "{:<10} {:>10x} {:>10x} {:>10x} {:>10x} {:>10x}",
            s.name,
            pe.rva_to_va(s.virtual_address),
            s.virtual_size,
            s.raw_address,
            s.raw_size,
            s.characteristics
        );
    }
    let imports = pe.imports();
    let mut mods: Vec<&str> = imports.iter().map(|(_, d, _)| d.as_str()).collect();
    mods.sort_unstable();
    mods.dedup();
    println!(
        "
imports: {} across {} modules",
        imports.len(),
        mods.len()
    );
    // TVM_IMPORT=<va> resolves a single IAT slot, for identifying the target of a
    // boxed call seen in a disassembly.
    if let Ok(q) = std::env::var("TVM_IMPORT") {
        let want = u64::from_str_radix(q.trim_start_matches("0x"), 16).unwrap_or(0);
        match imports.iter().find(|(slot, _, _)| *slot == want) {
            Some((slot, dll, sym)) => println!("  [{slot:#x}] {dll}!{sym}"),
            None => println!("  [{want:#x}] not an import slot"),
        }
    }
    if std::env::var_os("TVM_ALL_IMPORTS").is_some() {
        for (slot, dll, sym) in imports.iter() {
            println!("  [{slot:#x}] {dll}!{sym}");
        }
    }
    for (slot, dll, sym) in imports.iter().take(6) {
        println!("  [{slot:#x}] {dll}!{sym}");
    }

    match pe.section_by_name(vm_section) {
        Some(s) => println!(
            "\nVM section {vm_section} present: {:#x}..{:#x} ({:.2} MiB)",
            pe.rva_to_va(s.virtual_address),
            pe.rva_to_va(s.virtual_address) + s.virtual_size as u64,
            s.virtual_size as f64 / (1024.0 * 1024.0)
        ),
        None => println!("\nVM section {vm_section} NOT found"),
    }
    Ok(())
}

pub fn cmd_entries(path: &PathBuf, vm_section: &str, all: bool) -> Result<()> {
    let pe = pe::PeFile::load(path)?;
    let entries = discover::find_vm_entries(&pe, vm_section);
    println!("discovered {} virtualized functions", entries.len());
    let show = if all {
        entries.len()
    } else {
        entries.len().min(16)
    };
    for e in &entries[..show] {
        let sec = pe
            .section_for_va(e.trampoline_va)
            .map(|s| s.name.clone())
            .unwrap_or_default();
        println!(
            "  {:#x} [{}] -> {:#x}  (int3 pad {})",
            e.trampoline_va, sec, e.vm_entry_va, e.int3_padding
        );
    }
    if show < entries.len() {
        println!("  ... {} more (use --all)", entries.len() - show);
    }
    if pe.entry_point_rva != 0 {
        let ep = pe.rva_to_va(pe.entry_point_rva);
        if let Some(e) = entries.iter().find(|e| e.trampoline_va == ep) {
            println!(
                "\nmodule entry point is virtualized: {:#x} -> {:#x}",
                ep, e.vm_entry_va
            );
        }
    }
    Ok(())
}

pub fn cmd_dis(path: &PathBuf, va: u64, count: usize) -> Result<()> {
    let pe = pe::PeFile::load(path)?;
    let insts = disasm::decode_run(&pe, va, count);
    anyhow::ensure!(!insts.is_empty(), "nothing decodable at {va:#x}");
    let mut fmt = IntelFormatter::new();
    let mut s = String::new();
    for i in &insts {
        s.clear();
        fmt.format(i, &mut s);
        let bytes = pe.read_va(i.ip(), i.len()).context("bytes")?;
        println!(
            "{:>12x}  {:<24} {}",
            i.ip(),
            bytes.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            s
        );
    }
    Ok(())
}
