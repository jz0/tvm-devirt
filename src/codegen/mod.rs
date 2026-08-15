//! x86 code generation.

use anyhow::{Result, bail};
use iced_x86::code_asm::*;
use std::collections::{HashMap, HashSet};

use crate::binary::pe::PeFile;
use crate::ir::expr::{Reg, Width};
use crate::ir::regalloc::{self, Alloc, Item, Loc, Move};
use crate::ir::sched::{self, CallTarget, CastKind, Instr, Operand, SchedBlock};
use crate::ir::{BlockId, Cfg, Terminator};

mod control;
mod emit;
mod function;
mod layout;
mod operands;
mod unwind;

pub(crate) use control::*;
pub(crate) use emit::*;
pub(crate) use function::*;
pub(crate) use layout::*;
pub(crate) use operands::*;
pub(crate) use unwind::*;

#[cfg(test)]
mod tests;
