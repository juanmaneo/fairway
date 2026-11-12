//! x86_64 SIMD backends: AVX2 and SSE4.1 (ports of `hh_avx2.h`/`hh_sse41.h`).
//!
//! All intrinsics are wrapped in functions that are only called after
//! runtime feature detection, so callers must uphold that invariant.

pub mod avx2;
pub mod sse41;

pub(crate) use super::shared::{load3_allow_read_before_and_return, load3_allow_unordered};

/// Runtime detection needs `std`; without it, only features enabled at compile
/// time (e.g. `-C target-feature=+avx2`) are used.
#[inline]
pub fn has_sse41() -> bool {
    #[cfg(feature = "std")]
    {
        std::arch::is_x86_feature_detected!("sse4.1")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "sse4.1")
    }
}

#[inline]
pub fn has_avx2() -> bool {
    #[cfg(feature = "std")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx2")
    }
}
