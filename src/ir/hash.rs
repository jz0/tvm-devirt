//! Hashing helpers

pub(super) const FNV1A_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_PRIME: u64 = 0x100_0000_01b3;

pub(super) fn mix_bytes(bytes: &[u8], hash: &mut u64) {
    for &byte in bytes {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(FNV1A_PRIME);
    }
}

pub(super) fn mix_u64(value: u64, hash: &mut u64) {
    mix_bytes(&value.to_le_bytes(), hash);
}
