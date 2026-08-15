//! Commands that dump an intermediate stage of the pipeline: the recovered
//! CFG, its SSA form, and the scheduled machine-level IR.
//!
//! Diagnostics only. Nothing here is on the writeback path.

use crate::cli::fmt::resolve_start;
use crate::{binary::pe, ir, ir::expr, ir::lift, ir::regalloc, ir::sched, vm::explore};
use anyhow::Result;
use std::path::PathBuf;

/// Build an `Explorer` with the shared CLI recovery settings.
fn explorer<'a>(
    pe: &'a pe::PeFile,
    stack_base: u64,
    steps: usize,
    max_blocks: usize,
    timeout_secs: u64,
    vm_section: &str,
) -> explore::Explorer<'a> {
    let mut ex = explore::Explorer::with_vm_section(pe, stack_base, vm_section);
    ex.step_budget = steps;
    ex.block_budget = max_blocks;
    ex.time_budget = std::time::Duration::from_secs(timeout_secs);
    ex
}

pub fn cmd_cfg(
    path: &PathBuf,
    va: u64,
    stack_base: u64,
    steps: usize,
    max_blocks: usize,
    show_events: bool,
    show_state: bool,
    timeout_secs: u64,
    vm_section: &str,
) -> Result<()> {
    let pe = pe::PeFile::load(path)?;
    let start = resolve_start(&pe, va);
    if start != va {
        println!("trampoline {va:#x} -> {start:#x}");
    }

    let mut ex = explorer(&pe, stack_base, steps, max_blocks, timeout_secs, vm_section);
    ex.verbose = true;

    let t0 = std::time::Instant::now();
    let cfg = ex.recover(start);
    let elapsed = t0.elapsed();

    println!(
        "recovered {} blocks in {:.1?} ({} VM instructions interpreted)",
        cfg.blocks.len(),
        elapsed,
        cfg.total_steps
    );
    let s = ir::summarize(&cfg);
    let mut kinds: Vec<_> = s.iter().collect();
    kinds.sort();
    println!(
        "terminators: {}",
        kinds
            .iter()
            .map(|(k, v)| format!("{v} {k}"))
            .collect::<Vec<_>>()
            .join(", ")
    );

    for b in &cfg.blocks {
        let ev = b.events.len();
        print!(
            "  {:<28} cost {:<6} events {:<3} ",
            b.id.to_string(),
            b.cost,
            ev
        );
        match &b.terminator {
            ir::Terminator::Jump(t) => println!("jmp {t}"),
            ir::Terminator::Branch {
                taken,
                not_taken,
                predicate,
            } => {
                println!(
                    "branch [{}] ? {taken} : {not_taken}",
                    expr::render(&cfg.arena, *predicate, 4)
                )
            }
            ir::Terminator::Switch { selector, targets } => {
                println!(
                    "switch [{}] over {} arms",
                    expr::render(&cfg.arena, *selector, 4),
                    targets.len()
                );
                for t in targets {
                    println!("        -> {t}");
                }
            }
            ir::Terminator::Return { dest } => {
                println!("ret to {}", expr::render(&cfg.arena, *dest, 4))
            }
            ir::Terminator::TailCall { target } => println!("tailcall {target:#x}"),
            ir::Terminator::Unresolved { reason } => println!("UNRESOLVED: {reason}"),
            ir::Terminator::Backedge { target } => println!("backedge -> {target}"),
            ir::Terminator::Adopted { taken, not_taken } => {
                println!("branch [predicate not re-derived] ? {taken} : {not_taken}")
            }
            ir::Terminator::AdoptedReturn => println!("ret (dest not re-derived)"),
            ir::Terminator::AdoptedSwitch { targets } => println!(
                "switch [selector not re-derived] -> {}",
                targets
                    .iter()
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
        if show_state {
            for (r, v) in &b.exit_regs {
                let rendered = expr::render(&cfg.arena, *v, 4);
                // Only show registers holding something other than their entry
                // value; the rest is noise.
                if rendered != format!("{r:?}").to_lowercase() {
                    println!("        {r:?} = {rendered}");
                }
            }
        }
        if show_events {
            for e in &b.events {
                match e {
                    lift::Event::Store {
                        addr,
                        value,
                        width,
                        site,
                        region,
                    } if std::env::var("TVM_FULL_ADDR").is_ok() => {
                        println!(
                            "      store{:<3} {:<10} [{}] <- {} @{site:#x}",
                            width.bits(),
                            format!("{region:?}"),
                            expr::render(&cfg.arena, *addr, 60),
                            expr::render(&cfg.arena, *value, 4)
                        )
                    }
                    lift::Event::Store {
                        addr,
                        value,
                        width,
                        site,
                        region,
                    } => println!(
                        "      store{:<3} {:<10} [{}] <- {} @{site:#x}",
                        width.bits(),
                        format!("{region:?}"),
                        expr::render(&cfg.arena, *addr, 3),
                        expr::render(&cfg.arena, *value, 3)
                    ),
                    lift::Event::Load {
                        addr,
                        width,
                        site,
                        region,
                        ..
                    } => println!(
                        "      load{:<4} {:<10} [{}] @{site:#x}",
                        width.bits(),
                        format!("{region:?}"),
                        expr::render(
                            &cfg.arena,
                            *addr,
                            if std::env::var("TVM_FULL_ADDR").is_ok() {
                                60
                            } else {
                                3
                            }
                        )
                    ),
                    lift::Event::Call { site, .. } => println!("      call @{site:#x}"),
                    lift::Event::Boxed { site, text, .. } => {
                        println!("      boxed @{site:#x} {text}")
                    }
                }
            }
        }
    }

    if cfg.unresolved > 0 {
        println!("\n{} block(s) unresolved", cfg.unresolved);
    }
    Ok(())
}

/// Dump the VM context so the guest register image can be located.
#[allow(clippy::too_many_arguments)]
pub fn cmd_ssa(
    path: &PathBuf,
    va: u64,
    stack_base: u64,
    steps: usize,
    max_blocks: usize,
    timeout: u64,
    joins_only: bool,
    vm_section: &str,
) -> Result<()> {
    let pe = pe::PeFile::load(path)?;
    let start = resolve_start(&pe, va);
    let ex = explorer(&pe, stack_base, steps, max_blocks, timeout, vm_section);
    let cfg = ex.recover(start);

    println!("{} blocks, entry {}", cfg.blocks.len(), cfg.entry);
    for b in &cfg.blocks {
        if b.params.is_empty() {
            continue;
        }
        println!(
            "  {} = {} ({} preds, {} params)",
            b.block_ref,
            b.id,
            b.preds.len(),
            b.params.len()
        );
    }

    let phis = cfg.phis();
    println!(
        "
{} phi nodes",
        phis.len()
    );
    let mut shown = 0usize;
    for phi in &phis {
        // Distinct incoming values are what make a phi meaningful; a phi whose
        // predecessors all supply the same value is just a copy.
        let mut distinct: Vec<expr::Ref> = phi.incoming.iter().filter_map(|(_, v)| *v).collect();
        distinct.sort();
        distinct.dedup();
        if joins_only && distinct.len() < 2 {
            continue;
        }
        println!("  {}.{} <- ", phi.block, phi.reg.name());
        for (pred, v) in &phi.incoming {
            match v {
                Some(v) => println!("      {pred}: {}", expr::render(&cfg.arena, *v, 4)),
                None => println!("      {pred}: <predecessor not recovered>"),
            }
        }
        shown += 1;
    }
    if joins_only {
        println!("({shown} with >1 distinct incoming value)");
    }

    Ok(())
}

/// Schedule a recovered function and print the resulting three-address code.
pub fn cmd_lower(
    path: &PathBuf,
    va: u64,
    stack_base: u64,
    steps: usize,
    max_blocks: usize,
    timeout: u64,
    vm_section: &str,
) -> Result<()> {
    let pe = pe::PeFile::load(path)?;
    let start = resolve_start(&pe, va);
    let ex = explorer(&pe, stack_base, steps, max_blocks, timeout, vm_section);
    let cfg = ex.recover(start);

    let imports: std::collections::HashMap<u64, String> = pe
        .imports()
        .into_iter()
        .map(|(slot, dll, sym)| (slot, format!("{dll}!{sym}")))
        .collect();

    let blocks = sched::schedule(&cfg);
    println!(
        "{} blocks, {} instructions
",
        blocks.len(),
        sched::instr_count(&blocks)
    );
    for b in &blocks {
        print!("{}", sched::render_block_with(b, &imports));
        if std::env::var_os("TVM_SHOW_ALLOC").is_some() {
            let a = regalloc::allocate(b);
            let mut segs = a.segments.clone();
            segs.sort_by_key(|s| (s.start, s.end));
            for sg in &segs {
                println!(
                    "      seg {:?} [{}..{}) -> {:?}",
                    sg.item, sg.start, sg.end, sg.loc
                );
            }
            for c in &a.copies {
                println!(
                    "      copy {:?} at {} {:?} -> {:?}",
                    c.item, c.at, c.from, c.to
                );
            }
            for m in regalloc::reconcile(b, &a) {
                println!("      {}", regalloc::render_move(&m));
            }
            for e in regalloc::verify(b, &a) {
                println!("      ERR {e:?}");
            }
        }
        match &b.terminator {
            ir::Terminator::Jump(t) => println!("    jmp {t}"),
            ir::Terminator::Branch {
                taken, not_taken, ..
            } => {
                println!("    br ? {taken} : {not_taken}")
            }
            ir::Terminator::Switch { targets, .. } => println!("    switch {} arms", targets.len()),
            ir::Terminator::Return { .. } => println!("    ret"),
            ir::Terminator::TailCall { target } => println!("    tail {target:#x}"),
            ir::Terminator::Backedge { target } => println!("    loop -> {target}"),
            ir::Terminator::Adopted { taken, not_taken } => {
                println!("    br(adopted) ? {taken} : {not_taken}")
            }
            ir::Terminator::AdoptedSwitch { targets } => {
                println!("    switch(adopted) {} arms", targets.len())
            }
            ir::Terminator::AdoptedReturn => println!("    ret(adopted)"),
            ir::Terminator::Unresolved { reason } => {
                println!("    UNRESOLVED: {}", &reason[..reason.len().min(90)])
            }
        }
        println!();
    }
    Ok(())
}
