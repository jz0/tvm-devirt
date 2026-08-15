//! Commands that emit one recovered function or write a patched binary.

use crate::cli::fmt::resolve_start;
use crate::{
    binary::pe, binary::writeback, codegen, ir, ir::regalloc, ir::sched, vm::discover, vm::explore,
};
use anyhow::Result;
use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};

fn default_jobs() -> usize {
    const MAX_DEFAULT_JOBS: usize = 8;
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(MAX_DEFAULT_JOBS)
}

pub fn cmd_write_devirt(
    input: &std::path::Path,
    output: &std::path::Path,
    steps: usize,
    timeout: u64,
    vm_section: &str,
    jobs: Option<usize>,
) -> Result<()> {
    let pe = pe::PeFile::load(input)?;
    let entries = discover::find_vm_entries(&pe, vm_section);
    let jobs = jobs.unwrap_or_else(default_jobs).max(1);
    eprintln!(
        "devirtualizing {} functions… ({jobs} at a time)",
        entries.len()
    );
    let (out, report) = writeback::write_devirt(&pe, &entries, steps, timeout, vm_section, jobs)?;
    std::fs::write(output, &out)?;
    println!("wrote {} bytes to {:?}", out.len(), output);

    println!(
        "retargeted {}/{} functions",
        report.retargeted,
        report.total()
    );

    if !report.incomplete.is_empty() {
        println!("  {} failed to recover:", report.incomplete.len());
        let mut by_reason: std::collections::BTreeMap<&str, Vec<u64>> =
            std::collections::BTreeMap::new();
        for (va, reason) in &report.incomplete {
            by_reason.entry(reason.as_str()).or_default().push(*va);
        }
        for (reason, vas) in by_reason {
            println!("    {}: {reason}", fmt_vas(&vas));
        }
    }

    if !report.failed.is_empty() {
        println!("  {} recovered but failed to lower:", report.failed.len());
        let mut by_reason: std::collections::BTreeMap<&str, Vec<u64>> =
            std::collections::BTreeMap::new();
        for (va, reason) in &report.failed {
            by_reason.entry(reason.as_str()).or_default().push(*va);
        }
        for (reason, vas) in by_reason {
            println!("    {}: {reason}", fmt_vas(&vas));
        }
    }

    println!(
        "  unwind: {} RUNTIME_FUNCTION records ({} new), exception dir updated",
        report.pdata_entries, report.pdata_added
    );
    if report.pdata_added != report.retargeted {
        println!(
            "    WARNING: {} functions retargeted but got only {} unwind records",
            report.retargeted, report.pdata_added
        );
    }
    println!(
        "  checksum: {:#010x} sig: {}",
        report.checksum, report.dropped_signature
    );
    Ok(())
}

/// Format a list of addresses, truncated so one bad pass cannot bury the summary.
fn fmt_vas(vas: &[u64]) -> String {
    const MAX: usize = 8;
    let shown: Vec<String> = vas.iter().take(MAX).map(|v| format!("{v:#x}")).collect();
    if vas.len() > MAX {
        format!("{} … and {} more", shown.join(" "), vas.len() - MAX)
    } else {
        shown.join(" ")
    }
}

/// Emit x86 machine code for one function and write it, or hex-dump to stdout.
pub fn cmd_devirt(
    path: &std::path::Path,
    va: u64,
    output: Option<&std::path::Path>,
    stack_base: u64,
    steps: usize,
    timeout: u64,
    dis: bool,
) -> Result<()> {
    let pe = pe::PeFile::load(path)?;
    let start = resolve_start(&pe, va);
    let mut ex = explore::Explorer::new(&pe, stack_base);
    ex.step_budget = steps;
    ex.time_budget = std::time::Duration::from_secs(timeout);
    let cfg = ex.recover(start);

    let unresolved = cfg
        .blocks
        .iter()
        .filter(|b| matches!(&b.terminator, ir::Terminator::Unresolved { .. }))
        .count();
    println!(
        "recovered {} blocks ({} unresolved) in {:.0?}",
        cfg.blocks.len(),
        unresolved,
        cfg.total_steps,
    );

    let sched_blocks = sched::schedule(&cfg);
    let allocs: Vec<regalloc::Alloc> = sched_blocks.iter().map(regalloc::allocate).collect();

    let bytes = codegen::emit_function(&cfg, &sched_blocks, &allocs, start, &pe)?;

    println!("emitted {} bytes at base {:#x}", bytes.len(), start);
    if dis {
        let mut decoder = Decoder::with_ip(64, &bytes, start, DecoderOptions::NONE);
        let mut formatter = IntelFormatter::new();
        formatter.options_mut().set_hex_prefix("0x");
        formatter.options_mut().set_hex_suffix("");
        for instr in &mut decoder {
            let mut s = String::new();
            formatter.format(&instr, &mut s);
            println!("  {:#x}  {}", instr.ip(), s);
        }
        println!();
    }

    match output {
        Some(p) => {
            std::fs::write(p, &bytes)?;
            println!("wrote {:?}", p);
        }
        None => {
            // Hex dump — 16 bytes per line with address prefix.
            for (i, chunk) in bytes.chunks(16).enumerate() {
                let addr = start + (i * 16) as u64;
                let hex: Vec<String> = chunk.iter().map(|b| format!("{b:02x}")).collect();
                println!("{addr:#x}:  {}", hex.join(" "));
            }
        }
    }
    Ok(())
}
