//! x86_64 SIMD backends: AVX2 and SSE4.1 (ports of `hh_avx2.h`/`hh_sse41.h`),
//! plus an AVX-512VL re-encoding of the AVX2 algorithm behind the `avx512`
//! cargo feature (there is no upstream AVX-512 file to port).
//!
//! All intrinsics are wrapped in functions that are only called after
//! runtime feature detection, so callers must uphold that invariant.

pub mod avx2;
#[cfg(feature = "avx512")]
pub mod avx512;
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

/// The `avx512` backend runs on avx512f + avx512vl + avx512bw; every CPU with
/// VL and BW has F, so the three are detected together.
// The `avx512*` feature names were stabilized in Rust 1.89; this function only
// compiles with the `avx512` feature, which therefore requires 1.89+.
#[cfg(feature = "avx512")]
#[allow(clippy::incompatible_msrv)]
#[inline]
pub fn has_avx512() -> bool {
    #[cfg(feature = "std")]
    {
        std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512bw")
    }
    #[cfg(not(feature = "std"))]
    {
        cfg!(target_feature = "avx512f")
            && cfg!(target_feature = "avx512vl")
            && cfg!(target_feature = "avx512bw")
    }
}
