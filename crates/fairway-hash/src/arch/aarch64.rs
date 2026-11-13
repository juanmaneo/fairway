//! aarch64 SIMD backends, little-endian targets only: the NEON HighwayHash
//! state (a port of `hh_neon.h`) and a NEON SipTreeHash (new code; upstream
//! has none).
//!
//! NEON is always available on aarch64, so unlike x86 there is no runtime
//! feature detection. Both modules read input as little-endian lanes, so this
//! module is not built for big-endian aarch64, which uses the portable code.

pub(crate) mod neon;
pub(crate) mod neon_sip_tree;
