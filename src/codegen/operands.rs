use super::*;

/// Apply a width to a memory operand. This is for stores, where the memory operand is what carries the width: the value may be an immediate, which has no size of its own.
pub(crate) fn sized<M: Into<AsmMemoryOperand>>(mem: M, width: Width) -> AsmMemoryOperand {
    let mem = mem.into();
    match width {
        Width::W8 => byte_ptr(mem),
        Width::W16 => word_ptr(mem),
        Width::W32 => dword_ptr(mem),
        Width::W64 => qword_ptr(mem),
    }
}

pub(crate) fn qreg(r: Reg) -> AsmRegister64 {
    use Reg::*;
    match r {
        Rax => rax,
        Rcx => rcx,
        Rdx => rdx,
        Rbx => rbx,
        Rsp => rsp,
        Rbp => rbp,
        Rsi => rsi,
        Rdi => rdi,
        R8 => r8,
        R9 => r9,
        R10 => r10,
        R11 => r11,
        R12 => r12,
        R13 => r13,
        R14 => r14,
        R15 => r15,
    }
}

pub(crate) fn dreg(r: Reg) -> AsmRegister32 {
    use Reg::*;
    match r {
        Rax => eax,
        Rcx => ecx,
        Rdx => edx,
        Rbx => ebx,
        Rsp => esp,
        Rbp => ebp,
        Rsi => esi,
        Rdi => edi,
        R8 => r8d,
        R9 => r9d,
        R10 => r10d,
        R11 => r11d,
        R12 => r12d,
        R13 => r13d,
        R14 => r14d,
        R15 => r15d,
    }
}

/// The 16-bit view of a register, for the W16 cast forms.
pub(crate) fn wreg(r: Reg) -> AsmRegister16 {
    use Reg::*;
    match r {
        Rax => ax,
        Rcx => cx,
        Rdx => dx,
        Rbx => bx,
        Rsp => sp,
        Rbp => bp,
        Rsi => si,
        Rdi => di,
        R8 => r8w,
        R9 => r9w,
        R10 => r10w,
        R11 => r11w,
        R12 => r12w,
        R13 => r13w,
        R14 => r14w,
        R15 => r15w,
    }
}

pub(crate) fn breg(r: Reg) -> AsmRegister8 {
    use Reg::*;
    match r {
        Rax => al,
        Rcx => cl,
        Rdx => dl,
        Rbx => bl,
        Rsp => spl,
        Rbp => bpl,
        Rsi => sil,
        Rdi => dil,
        R8 => r8b,
        R9 => r9b,
        R10 => r10b,
        R11 => r11b,
        R12 => r12b,
        R13 => r13b,
        R14 => r14b,
        R15 => r15b,
    }
}

/// Resolve an operand to where it physically lives at instruction `pos`.
/// Returns None for immediates and Frame references, which the caller handles.
pub(crate) fn resolve(alloc: &Alloc, op: &Operand, pos: usize) -> Option<Loc> {
    match op {
        Operand::Val(v) => alloc.loc_at(Item::Val(*v), pos),
        Operand::Entry(r) | Operand::Param(r) => alloc
            .loc_at(Item::Saved(*r), pos)
            .or_else(|| alloc.loc_at(Item::Entry(*r), pos)),
        Operand::Imm(_) | Operand::Frame(_) | Operand::OutArg(_) => None,
    }
}
