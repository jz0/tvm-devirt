pub const DEFAULT_STACK_BASE: u64 = 0x7fff_fffe_fda0;
pub const DEFAULT_VM_SECTION: &str = ".tvm0";

pub mod discover;
pub mod explore;
pub mod state;
