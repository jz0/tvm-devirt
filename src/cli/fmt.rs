//! Formatting and address-resolution helpers shared by the CLI commands.

use crate::{binary::disasm, binary::pe, ir::expr, ir::lift};

/// Follow an entry trampoline into the VM section, if `va` is one.
pub fn resolve_start(pe: &pe::PeFile, va: u64) -> u64 {
    if let Some(inst) = disasm::decode_at(pe, va) {
        if inst.mnemonic() == iced_x86::Mnemonic::Jmp {
            if let Some(t) = disasm::direct_target(&inst) {
                return t;
            }
        }
    }
    va
}

pub fn describe_stop(emu: &lift::Emulator, stop: &lift::Stop) -> String {
    match stop {
        lift::Stop::SymbolicBranch { site, dest } => format!(
            "symbolic branch at {site:#x}, dest = {}",
            fmt_ref(&emu.arena, *dest)
        ),
        lift::Stop::NativeBranch {
            site,
            predicate,
            taken,
            not_taken,
        } => format!(
            "native branch at {site:#x} on {} -> {taken:#x} / {not_taken:#x}",
            fmt_ref(&emu.arena, *predicate)
        ),
        lift::Stop::Return { site, dest } => {
            format!("return at {site:#x} to {}", fmt_ref(&emu.arena, *dest))
        }
        lift::Stop::Unsupported { site, text, .. } => {
            format!("unsupported instruction at {site:#x}: {text}")
        }
        lift::Stop::Unreadable { site } => format!("unreadable code at {site:#x}"),
        lift::Stop::Backedge { site, target, .. } => {
            format!("loop back edge at {site:#x} -> {target:#x}")
        }
        lift::Stop::Budget { site } => format!("step budget exhausted at {site:#x}"),
        lift::Stop::Diverged { site, nodes } => {
            format!("folding diverged at {site:#x} ({nodes} DAG nodes)")
        }
        lift::Stop::OutOfImage { site } => format!("left the image at {site:#x}"),
        lift::Stop::NoReturn { site } => {
            format!("call to a noreturn import at {site:#x}")
        }
    }
}

/// One-line rendering of an expression, depth limited.
pub fn fmt_ref(a: &expr::Arena, r: expr::Ref) -> String {
    fmt_ref_depth(a, r, 4)
}

pub fn fmt_ref_depth(a: &expr::Arena, r: expr::Ref, max_depth: u32) -> String {
    fn go(a: &expr::Arena, r: expr::Ref, depth: u32, max: u32, out: &mut String) {
        use expr::Op;
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
                    expr::UnOp::Not => "~",
                    expr::UnOp::Neg => "-",
                    expr::UnOp::ParityByte => "parity",
                    expr::UnOp::Bswap => "bswap",
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
