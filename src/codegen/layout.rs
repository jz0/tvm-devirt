use super::*;
use iced_x86::{Code, Instruction};

#[derive(Clone, Copy)]
pub struct ImageRange {
    pub lo: u64,
    pub hi: u64,
}

impl ImageRange {
    /// An empty range, for tests that do not care about RIP-relative forms.
    #[cfg(test)]
    pub fn none() -> Self {
        Self::empty()
    }

    pub fn of(pe: &PeFile) -> Self {
        ImageRange {
            lo: pe.image_base,
            hi: pe.image_base + pe.size_of_image as u64,
        }
    }

    pub fn empty() -> Self {
        ImageRange { lo: 0, hi: 0 }
    }

    fn holds(&self, va: u64) -> bool {
        va >= self.lo && va < self.hi
    }

    /// Can an absolute access to `target` use a RIP-relative operand?
    pub(crate) fn rip_ok(&self, target: u64) -> bool {
        self.holds(target)
    }
}

/// Entry trampolines that have been replaced by devirtualized code.
#[derive(Default)]
pub struct CallRedirects {
    /// Trampoline VA -> devirtualized body VA.
    map: HashMap<u64, u64>,
}

impl CallRedirects {
    /// No redirects: every direct call keeps the target recovery produced.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, trampoline_va: u64, body_va: u64) {
        self.map.insert(trampoline_va, body_va);
    }

    /// The address a direct call to `va` should name.
    ///
    /// Unmapped targets are returned unchanged: a call to a function that was never
    /// virtualized, or to one whose recovery did not finish and so still needs its
    /// trampoline, must keep pointing where it did.
    pub(crate) fn resolve(&self, va: u64) -> u64 {
        self.map.get(&va).copied().unwrap_or(va)
    }

    /// Where `trampoline_va`'s body was placed, if it has one.
    ///
    /// `write_devirt` uses this to assert that the layout it probed and the layout it
    /// emits agree.
    pub fn body_va(&self, trampoline_va: u64) -> Option<u64> {
        self.map.get(&trampoline_va).copied()
    }
}

/// Build a RIP-relative memory operand naming an absolute target.
pub(crate) fn rip_at(target: u64) -> iced_x86::MemoryOperand {
    iced_x86::MemoryOperand::with_base_displ(iced_x86::Register::RIP, target as i64)
}

/// Emit `mov <reg>, [rip+target]` at the given width.
pub(crate) fn rip_load(asm: &mut CodeAssembler, dst: Reg, target: u64, width: Width) -> Result<()> {
    let code = match width {
        Width::W64 => Code::Mov_r64_rm64,
        Width::W32 => Code::Mov_r32_rm32,
        Width::W16 => Code::Mov_r16_rm16,
        Width::W8 => Code::Mov_r8_rm8,
    };
    asm.add_instruction(Instruction::with2(code, ireg(dst, width), rip_at(target))?)?;
    Ok(())
}

/// Emit `mov [rip+target], <reg>` at the given width.
pub(crate) fn rip_store_reg(
    asm: &mut CodeAssembler,
    target: u64,
    src: Reg,
    width: Width,
) -> Result<()> {
    let code = match width {
        Width::W64 => Code::Mov_rm64_r64,
        Width::W32 => Code::Mov_rm32_r32,
        Width::W16 => Code::Mov_rm16_r16,
        Width::W8 => Code::Mov_rm8_r8,
    };
    asm.add_instruction(Instruction::with2(code, rip_at(target), ireg(src, width))?)?;
    Ok(())
}

/// Emit `lea <reg>, [rip+target]`: the address of an image byte, computed relative to the instruction rather than written down.
pub(crate) fn rip_lea(asm: &mut CodeAssembler, dst: Reg, target: u64) -> Result<()> {
    asm.add_instruction(Instruction::with2(
        Code::Lea_r64_m,
        ireg(dst, Width::W64),
        rip_at(target),
    )?)?;
    Ok(())
}

/// Emit `mov [rip+target], imm` at the given width, or `None` if the immediate does
/// not fit the form (a W64 store takes only a sign-extended imm32).
pub(crate) fn rip_store_imm(
    asm: &mut CodeAssembler,
    target: u64,
    c: u64,
    width: Width,
) -> Result<bool> {
    let instr = match width {
        Width::W64 => {
            if (c as i64) < i32::MIN as i64 || (c as i64) > i32::MAX as i64 {
                return Ok(false);
            }
            Instruction::with2(Code::Mov_rm64_imm32, rip_at(target), c as i32)?
        }
        Width::W32 => Instruction::with2(Code::Mov_rm32_imm32, rip_at(target), c as u32)?,
        Width::W16 => Instruction::with2(Code::Mov_rm16_imm16, rip_at(target), c as u16 as u32)?,
        Width::W8 => Instruction::with2(Code::Mov_rm8_imm8, rip_at(target), c as u8 as u32)?,
    };
    asm.add_instruction(instr)?;
    Ok(true)
}

/// Map a guest `Reg` to its iced `Register` at the given width.
///
/// Replaces the four separate `ireg64`/`ireg32`/`ireg16`/`ireg8` helpers.
/// Used with `Instruction::with2` where width is a runtime value.
pub(crate) fn ireg(r: Reg, w: Width) -> iced_x86::Register {
    use Reg::*;
    use iced_x86::Register as R;
    match (r, w) {
        (Rax, Width::W64) => R::RAX,
        (Rax, Width::W32) => R::EAX,
        (Rax, Width::W16) => R::AX,
        (Rax, Width::W8) => R::AL,
        (Rcx, Width::W64) => R::RCX,
        (Rcx, Width::W32) => R::ECX,
        (Rcx, Width::W16) => R::CX,
        (Rcx, Width::W8) => R::CL,
        (Rdx, Width::W64) => R::RDX,
        (Rdx, Width::W32) => R::EDX,
        (Rdx, Width::W16) => R::DX,
        (Rdx, Width::W8) => R::DL,
        (Rbx, Width::W64) => R::RBX,
        (Rbx, Width::W32) => R::EBX,
        (Rbx, Width::W16) => R::BX,
        (Rbx, Width::W8) => R::BL,
        (Rsp, Width::W64) => R::RSP,
        (Rsp, Width::W32) => R::ESP,
        (Rsp, Width::W16) => R::SP,
        (Rsp, Width::W8) => R::SPL,
        (Rbp, Width::W64) => R::RBP,
        (Rbp, Width::W32) => R::EBP,
        (Rbp, Width::W16) => R::BP,
        (Rbp, Width::W8) => R::BPL,
        (Rsi, Width::W64) => R::RSI,
        (Rsi, Width::W32) => R::ESI,
        (Rsi, Width::W16) => R::SI,
        (Rsi, Width::W8) => R::SIL,
        (Rdi, Width::W64) => R::RDI,
        (Rdi, Width::W32) => R::EDI,
        (Rdi, Width::W16) => R::DI,
        (Rdi, Width::W8) => R::DIL,
        (R8, Width::W64) => R::R8,
        (R8, Width::W32) => R::R8D,
        (R8, Width::W16) => R::R8W,
        (R8, Width::W8) => R::R8L,
        (R9, Width::W64) => R::R9,
        (R9, Width::W32) => R::R9D,
        (R9, Width::W16) => R::R9W,
        (R9, Width::W8) => R::R9L,
        (R10, Width::W64) => R::R10,
        (R10, Width::W32) => R::R10D,
        (R10, Width::W16) => R::R10W,
        (R10, Width::W8) => R::R10L,
        (R11, Width::W64) => R::R11,
        (R11, Width::W32) => R::R11D,
        (R11, Width::W16) => R::R11W,
        (R11, Width::W8) => R::R11L,
        (R12, Width::W64) => R::R12,
        (R12, Width::W32) => R::R12D,
        (R12, Width::W16) => R::R12W,
        (R12, Width::W8) => R::R12L,
        (R13, Width::W64) => R::R13,
        (R13, Width::W32) => R::R13D,
        (R13, Width::W16) => R::R13W,
        (R13, Width::W8) => R::R13L,
        (R14, Width::W64) => R::R14,
        (R14, Width::W32) => R::R14D,
        (R14, Width::W16) => R::R14W,
        (R14, Width::W8) => R::R14L,
        (R15, Width::W64) => R::R15,
        (R15, Width::W32) => R::R15D,
        (R15, Width::W16) => R::R15W,
        (R15, Width::W8) => R::R15L,
    }
}

/// Where things live in the stack frame we synthesise. Layout (addresses grow upward, RSP points at the bottom): [rsp + 0]               spill slot 0 [rsp + 8]               spill slot 1 ...
pub struct FrameLayout {
    pub frame_size: u32,
    spill_slots: u32,
    /// Bytes reserved at the bottom of the frame for callees' shadow space.
    shadow: u32,
    /// Callee-saved registers this function writes, in save order. Each gets 8
    /// bytes just above the spill slots.
    saves: Vec<Reg>,
    /// Outgoing stack arguments the widest call in this function needs, beyond
    /// the four the ABI passes in registers.
    stack_args: u32,
    /// The image's VA range, used by `emit_mov_imm` to choose a RIP-relative `lea` instead of an absolute `mov r64, imm64` when the constant is an image address.
    pub img: ImageRange,
}

/// Scratch qwords above the spill slots. Slot 0 holds the scratch register while a definition whose home is a spill slot is computed (see [`with_spilled_dst`]).
pub(crate) const SCRATCH_SLOTS: i64 = 4;
pub(crate) const SCRATCH_BYTES: i64 = 8 * SCRATCH_SLOTS;

/// Win64 shadow space: every callee may write the 32 bytes above its return address.
pub(crate) const SHADOW: i64 = 32;

impl FrameLayout {
    /// The no-saves, no-stack-arguments case. Only the tests build a layout this
    /// plain; `emit_function` always goes through `full`.
    #[cfg(test)]
    pub fn new(frame_lo: i64, spill_slots: u32, has_calls: bool) -> Self {
        Self::with_saves(frame_lo, spill_slots, has_calls, Vec::new())
    }

    #[cfg(test)]
    pub fn with_saves(frame_lo: i64, spill_slots: u32, has_calls: bool, saves: Vec<Reg>) -> Self {
        Self::full(frame_lo, spill_slots, has_calls, saves, 0)
    }

    /// Reserve frame space for callee-saved registers and outgoing stack arguments.
    /// Saves sit above spills, while outgoing arguments sit above shadow space so the
    /// regions cannot alias.
    pub fn full(
        frame_lo: i64,
        spill_slots: u32,
        has_calls: bool,
        saves: Vec<Reg>,
        stack_args: u32,
    ) -> Self {
        // Bytes needed to cover spills + guest locals. Only offsets *below* entry RSP are local frame the prologue must reserve.
        let locals = frame_lo.min(0).unsigned_abs() as i64;
        // A function that calls must give its callee 32 bytes of shadow space at the very bottom of the frame, because the callee addresses those slots off its own RSP.
        let shadow = if has_calls { SHADOW } else { 0 };
        let raw = shadow
            + 8 * stack_args as i64
            + 8 * spill_slots as i64
            + SCRATCH_BYTES
            + 8 * saves.len() as i64
            + locals;
        // RSP % 16 == 8 at entry (call pushed the return address).
        // After sub rsp, N we want RSP % 16 == 0, so N ≡ 8 (mod 16).
        let aligned = ((raw + 15) & !15) as u32;
        let frame_size = if aligned % 16 == 8 {
            aligned
        } else {
            aligned + 8
        };
        FrameLayout {
            frame_size,
            spill_slots,
            shadow: shadow as u32,
            saves,
            stack_args,
            img: ImageRange::empty(),
        }
    }

    /// Displacement from current RSP of the slot preserving callee-saved `r`.
    pub fn save_disp(&self, r: Reg) -> Option<i64> {
        let i = self.saves.iter().position(|&s| s == r)?;
        Some(self.outgoing_end() + 8 * self.spill_slots as i64 + 8 * i as i64)
    }

    pub fn saves(&self) -> &[Reg] {
        &self.saves
    }

    /// Displacement from current RSP for a `Frame(o)` reference.
    pub fn frame_disp(&self, o: i64) -> i64 {
        // Frame(o) is entry_rsp + o.  After sub rsp, N, rsp = entry_rsp - N.
        // So the byte sits at (entry_rsp + o) - rsp = N + o.
        self.frame_size as i64 + o
    }

    /// Displacement from current RSP for `Spill(i)`.
    pub fn spill_disp(&self, i: u32) -> i64 {
        // Above the shadow space and the outgoing stack arguments, both of which a
        // callee addresses off its own RSP and would otherwise overwrite.
        self.outgoing_end() + i as i64 * 8
    }

    /// First byte above the shadow space and the outgoing stack arguments.
    fn outgoing_end(&self) -> i64 {
        self.shadow as i64 + 8 * self.stack_args as i64
    }

    /// Where the scratch register is parked while a spilled definition is computed.
    pub fn scratch_disp(&self) -> i64 {
        self.scratch_slot_disp(0)
    }

    /// Displacement of scratch slot `i`, `i < SCRATCH_SLOTS`. Slot 0 is [`scratch_disp`]; the rest belong to [`emit_rdx_rax_op`].
    pub fn scratch_slot_disp(&self, i: u32) -> i64 {
        debug_assert!((i as i64) < SCRATCH_SLOTS);
        self.outgoing_end()
            + self.spill_slots as i64 * 8
            + 8 * self.saves.len() as i64
            + 8 * i as i64
    }

    /// The byte ranges of the frame that only this emitter ever touches.
    ///
    /// Two disjoint half-open `[lo, hi)` displacement ranges from the current RSP:
    /// the spill slots and the scratch slots. Everything else in the frame is
    /// observable by something outside this function's own instruction stream:
    ///
    /// - shadow space and outgoing arguments are addressed by the *callee* off its
    ///   own RSP,
    /// - the callee-saved register area is read by the OS unwinder through the
    ///   unwind record, so a store there is live even when no instruction reads it,
    /// - guest locals (`Frame(o)`) can have their address taken and handed to a
    ///   callee, so a store may be read through a pointer not visible here.
    ///
    /// Spill and scratch slots have neither property: `spill_disp` and
    /// `scratch_slot_disp` are reached only through direct `[rsp+k]` operands this
    /// file emits, and nothing takes their address. That makes them the only part
    /// of the frame where "no later instruction reads this byte" is the same
    /// statement as "this store cannot be observed".
    ///
    /// The save area sits *between* the two ranges, which is why this returns two
    /// ranges rather than one span: covering the gap would let dead-store
    /// elimination delete a callee-saved register's spill.
    pub(crate) fn private_ranges(&self) -> [(i64, i64); 2] {
        let spills_lo = self.spill_disp(0);
        let spills_hi = spills_lo + 8 * self.spill_slots as i64;
        let scratch_lo = self.scratch_slot_disp(0);
        let scratch_hi = scratch_lo + 8 * SCRATCH_SLOTS;
        [(spills_lo, spills_hi), (scratch_lo, scratch_hi)]
    }

    /// Displacement of outgoing stack argument `n`, counting from 5 as the ABI
    /// does. Argument 5 is the first that does not travel in a register.
    /// Only the tests ask for this directly; `emit_call` computes the same
    /// displacement inline from `Operand::OutArg`.
    #[cfg(test)]
    pub fn stack_arg_disp(&self, n: u32) -> Option<i64> {
        if n < 5 || n - 4 > self.stack_args {
            return None;
        }
        Some(8 * (n as i64 - 1))
    }
}
