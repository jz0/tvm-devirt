mod binary;
mod cli;
mod codegen;
mod ir;
mod vm;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use cli::emit::{cmd_devirt, cmd_write_devirt};
use cli::inspect::{cmd_dis, cmd_entries, cmd_info, cmd_trace, cmd_vmctx};
use cli::recover::{cmd_cfg, cmd_lower, cmd_ssa};

#[derive(Parser)]
#[command(
    name = "tvm-devirt",
    about = "static devirtualizer for Tencent VM (TVM)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Dump PE layout
    Info {
        input: PathBuf,
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
    },
    /// List discovered virtualized function entry trampolines.
    Entries {
        input: PathBuf,
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
        /// Print all entries instead of a summary.
        #[arg(long)]
        all: bool,
    },
    /// Linear disassembly at a virtual address
    Dis {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, default_value_t = 60)]
        count: usize,
    },
    /// Guided symbolic evaluation of a virtualized function.
    Trace {
        input: PathBuf,
        /// Start VA: either the entry trampoline or a .tvm0 address.
        #[arg(value_parser = parse_u64)]
        va: u64,
        /// Instruction step budget.
        #[arg(long, default_value_t = 200_000)]
        steps: usize,
        /// Concrete value RSP is initialised to.
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        /// Print the guest events that were recorded.
        #[arg(long)]
        events: bool,
        /// Print the resolved VM handler transitions.
        #[arg(long)]
        dispatches: bool,
        /// Disassemble the last N executed instructions.
        #[arg(long, default_value_t = 0)]
        tail: usize,
        /// Stop at this VA and dump the register state (debugging aid).
        #[arg(long, value_parser = parse_u64)]
        break_at: Option<u64>,
    },
    /// Recover the control flow graph of a virtualized function.
    Cfg {
        input: PathBuf,
        /// Start VA: entry trampoline or a .tvm0 address.
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        /// Per-path instruction budget.
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        /// Maximum blocks to recover.
        #[arg(long, default_value_t = 512)]
        max_blocks: usize,
        /// Print each recovered block's guest events.
        #[arg(long)]
        events: bool,
        /// Print each block's exit register state.
        #[arg(long)]
        state: bool,
        /// Wall-clock limit in seconds for the whole recovery.
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Section holding VM bytecode.
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
    },
    /// Dump the VM context window around RBP, to locate the guest register image.
    Vmctx {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        /// Stop after this many VM instructions & dump.
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        /// Bytes of context to dump either side of RBP.
        #[arg(long, default_value_t = 0x120)]
        window: u64,
    },
    /// Show SSA parameters and phi nodes for a recovered function.
    Ssa {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        max_blocks: usize,
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Only show phis with more than one distinct incoming value.
        #[arg(long)]
        joins_only: bool,
        /// Section holding VM bytecode.
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
    },
    /// Schedule a recovered function into linear three-address code.
    Lower {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        max_blocks: usize,
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Section holding VM bytecode.
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
    },
    /// Devirtualize all functions and write a patched binary.
    WriteDevirt {
        input: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 500_000)]
        steps: usize,
        /// Wall-clock backstop per function.
        #[arg(long, default_value_t = 600)]
        timeout: u64,
        /// Section holding VM bytecode.
        #[arg(long, default_value = vm::DEFAULT_VM_SECTION)]
        vm_section: String,
        /// Functions to recover concurrently. Defaults to `min(cores, 8)`.
        #[arg(long)]
        jobs: Option<usize>,
    },
    /// Devirtualize one function into x86 machine code.
    Devirt {
        input: PathBuf,
        #[arg(value_parser = parse_u64)]
        va: u64,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, value_parser = parse_u64, default_value_t = vm::DEFAULT_STACK_BASE)]
        stack_base: u64,
        #[arg(long, default_value_t = 400_000)]
        steps: usize,
        #[arg(long, default_value_t = 120)]
        timeout: u64,
        /// Disassemble the emitted bytes after encoding.
        #[arg(long)]
        dis: bool,
    },
}

fn parse_u64(s: &str) -> Result<u64, String> {
    let t = s.trim();
    let r = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .map_or_else(|| t.parse::<u64>(), |h| u64::from_str_radix(h, 16));
    r.map_err(|e| e.to_string())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info { input, vm_section } => cmd_info(&input, &vm_section),
        Cmd::Entries {
            input,
            vm_section,
            all,
        } => cmd_entries(&input, &vm_section, all),
        Cmd::Dis { input, va, count } => cmd_dis(&input, va, count),
        Cmd::Trace {
            input,
            va,
            steps,
            stack_base,
            events,
            dispatches,
            tail,
            break_at,
        } => cmd_trace(
            &input, va, steps, stack_base, events, dispatches, tail, break_at,
        ),
        Cmd::Cfg {
            input,
            va,
            stack_base,
            steps,
            max_blocks,
            events,
            state,
            timeout,
            vm_section,
        } => cmd_cfg(
            &input,
            va,
            stack_base,
            steps,
            max_blocks,
            events,
            state,
            timeout,
            &vm_section,
        ),
        Cmd::Vmctx {
            input,
            va,
            stack_base,
            steps,
            window,
        } => cmd_vmctx(&input, va, stack_base, steps, window),
        Cmd::Ssa {
            input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            joins_only,
            vm_section,
        } => cmd_ssa(
            &input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            joins_only,
            &vm_section,
        ),
        Cmd::WriteDevirt {
            input,
            output,
            steps,
            timeout,
            vm_section,
            jobs,
        } => cmd_write_devirt(&input, &output, steps, timeout, &vm_section, jobs),
        Cmd::Devirt {
            input,
            va,
            output,
            stack_base,
            steps,
            timeout,
            dis,
        } => cmd_devirt(
            &input,
            va,
            output.as_deref(),
            stack_base,
            steps,
            timeout,
            dis,
        ),
        Cmd::Lower {
            input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            vm_section,
        } => cmd_lower(
            &input,
            va,
            stack_base,
            steps,
            max_blocks,
            timeout,
            &vm_section,
        ),
    }
}
