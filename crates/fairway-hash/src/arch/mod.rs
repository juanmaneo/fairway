//! Architecture-specific code. Shared byte-loading helpers live here so both
//! x86 and aarch64 backends use identical length-padding semantics.

#[cfg(target_arch = "x86_64")]
pub mod x86;

/// Shared `Load3` helpers (from `load3.h`). These are `unsafe` because the
/// "read before" policy may read up to 4 bytes before the returned range;
/// callers guarantee those bytes are within the input buffer.
pub mod shared {
    /// Loads 0..=3 bytes; the bytes need not be in little-endian order.
    /// This order is part of the HighwayHash length-padding definition.
    #[inline(always)]
    pub unsafe fn load3_allow_unordered(from: *const u8, size_mod4: usize) -> u64 {
        if size_mod4 == 0 {
            return 0;
        }
        let idx1 = size_mod4 >> 1;
        let idx2 = size_mod4 - 1;
        // SAFETY: the caller guarantees `from..from+size_mod4` is readable.
        unsafe {
            (*from.add(0)) as u64
                + (((*from.add(idx1)) as u64) << 8)
                + (((*from.add(idx2)) as u64) << 16)
        }
    }

    /// Up to 4 preceding bytes may be read and returned; valid bytes are in
    /// little-endian order in the low bytes. Reads 4 bytes ending at
    /// `from + size_mod4`; caller guarantees they are in bounds.
    #[inline(always)]
    pub unsafe fn load3_allow_read_before_and_return(from: *const u8, size_mod4: usize) -> u32 {
        // SAFETY: the caller guarantees the 4 bytes ending at `from + size_mod4`
        // are readable (they may start up to 4 bytes before `from`).
        let p = unsafe { from.add(size_mod4).sub(4) };
        // SAFETY: same guarantee as above.
        u32::from_le_bytes(unsafe { [*p, *p.add(1), *p.add(2), *p.add(3)] })
    }
}
