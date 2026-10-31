//! Rust port of Google's [HighwayHash](https://github.com/google/highwayhash)
//! and its companion SipHash implementation.
//!
//! HighwayHash is a strong pseudorandom function with security claims
//! (<https://arxiv.org/abs/1612.06257>), about 4x faster than SipHash and 10x
//! faster than BLAKE2. This crate reproduces the exact, frozen output of the
//! upstream C++ library (verified against its golden test vectors) and picks
//! the best available implementation at runtime:
//!
//! * AVX2 and SSE4.1 on x86_64 (via `core::arch`, detected with `std::arch`),
//! * NEON on aarch64,
//! * a portable scalar fallback everywhere else (also covers big-endian).
//!
//! # Examples
//!
//! ```
//! use fairway::{HighwayHasher, Key};
//!
//! let key = Key([1, 2, 3, 4]);
//! let mut h = HighwayHasher::new(key);
//! h.update(b"hello ");
//! h.update(b"world");
//! let hash64: u64 = h.finalize64();
//! let hash128: [u64; 2] = h.finalize128();
//! let hash256: [u64; 4] = h.finalize256();
//! ```
//!
//! One-shot helpers use the best implementation for the current CPU:
//!
//! ```
//! use fairway::{fairway64, Key};
//! assert_eq!(fairway64(Key([1, 2, 3, 4]), b"abc"), fairway64(Key([1, 2, 3, 4]), b"abc"));
//! ```

// Unit tests use `std` (the test harness needs it anyway), even without the
// `std` feature.
#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
// SIMD backends use `core::arch` intrinsics, which are `unsafe`; the unsafe
// code is confined to the `x86`/`neon` modules and only invoked after runtime
// CPU feature detection (or explicit `target_feature` callers).

#[cfg(all(
    any(target_arch = "x86_64", target_arch = "aarch64"),
    not(feature = "portable-only")
))]
mod arch;
#[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
mod neon;
mod portable;
mod sip;
#[doc(hidden)]
#[rustfmt::skip] // Keep the golden tables compact, as transcribed.
pub mod vectors;

#[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
pub use arch::x86::{avx2, sse41};

pub use sip::{sip_hash, sip_hash13, SipHashKey};

/// 256-bit secret key. Keep it unknown to attackers; initialize it randomly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Key(pub [u64; 4]);

impl From<[u64; 4]> for Key {
    fn from(k: [u64; 4]) -> Self {
        Key(k)
    }
}

/// Which backend produced a hash / is available on this machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    Avx2,
    Sse41,
    Neon,
    Portable,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Avx2 => "AVX2",
            Backend::Sse41 => "SSE4.1",
            Backend::Neon => "NEON",
            Backend::Portable => "Portable",
        }
    }
}

/// Best backend available on the current CPU (runtime-detected on x86_64).
pub fn best_backend() -> Backend {
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
    {
        if arch::x86::has_avx2() {
            return Backend::Avx2;
        }
        if arch::x86::has_sse41() {
            return Backend::Sse41;
        }
    }
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
    {
        Backend::Neon
    }
    #[cfg(not(all(target_arch = "aarch64", not(feature = "portable-only"))))]
    {
        Backend::Portable
    }
}

enum Inner {
    Portable(portable::State),
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
    Avx2(arch::x86::avx2::State),
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
    Sse41(arch::x86::sse41::State),
    #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
    Neon(neon::State),
}

impl Inner {
    fn new(key: Key, backend: Backend) -> Self {
        match backend {
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Backend::Avx2 if arch::x86::has_avx2() => {
                // SAFETY: AVX2 support was just detected.
                Inner::Avx2(unsafe { arch::x86::avx2::State::new(key) })
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Backend::Sse41 if arch::x86::has_sse41() => {
                // SAFETY: SSE4.1 support was just detected.
                Inner::Sse41(unsafe { arch::x86::sse41::State::new(key) })
            }
            #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
            Backend::Neon => Inner::Neon(neon::State::new(key)),
            // Portable, or a backend this CPU/build cannot run.
            _ => Inner::Portable(portable::State::new(key)),
        }
    }

    fn update_packet(&mut self, packet: &[u8; 32]) {
        match self {
            Inner::Portable(s) => s.update_packet(packet),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => unsafe { s.update_packet(packet) },
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => unsafe { s.update_packet(packet) },
            #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
            Inner::Neon(s) => s.update_packet(packet),
        }
    }

    fn update_remainder(&mut self, bytes: &[u8]) {
        match self {
            Inner::Portable(s) => s.update_remainder(bytes),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => unsafe { s.update_remainder(bytes) },
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => unsafe { s.update_remainder(bytes) },
            #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
            Inner::Neon(s) => s.update_remainder(bytes),
        }
    }

    fn finalize64(&mut self) -> u64 {
        match self {
            Inner::Portable(s) => s.finalize64(),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => unsafe { s.finalize64() },
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => unsafe { s.finalize64() },
            #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
            Inner::Neon(s) => s.finalize64(),
        }
    }

    fn finalize128(&mut self) -> [u64; 2] {
        match self {
            Inner::Portable(s) => s.finalize128(),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => unsafe { s.finalize128() },
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => unsafe { s.finalize128() },
            #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
            Inner::Neon(s) => s.finalize128(),
        }
    }

    fn finalize256(&mut self) -> [u64; 4] {
        match self {
            Inner::Portable(s) => s.finalize256(),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => unsafe { s.finalize256() },
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => unsafe { s.finalize256() },
            #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
            Inner::Neon(s) => s.finalize256(),
        }
    }

    fn clone_inner(&self) -> Inner {
        match self {
            Inner::Portable(s) => Inner::Portable(s.clone()),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => Inner::Avx2(*s),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => Inner::Sse41(*s),
            #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
            Inner::Neon(s) => Inner::Neon(*s),
        }
    }
}

/// Incremental HighwayHash hasher (equivalent of `HighwayHashCatT`).
///
/// Uses the best backend for the current CPU. `update` may be called any
/// number of times; the result equals hashing the concatenation of all
/// inputs. Note that hashing `["A", ""]` equals `["", "A"]`; append a length
/// or non-empty marker per field if that matters.
pub struct HighwayHasher {
    inner: Inner,
    buffer: [u8; 32],
    buffer_usage: usize,
}

impl HighwayHasher {
    pub fn new(key: Key) -> Self {
        Self::with_backend(key, best_backend())
    }

    /// Uses `backend` if this CPU/build supports it, otherwise the portable
    /// one; [`HighwayHasher::backend`] reports which one is in use.
    pub fn with_backend(key: Key, backend: Backend) -> Self {
        HighwayHasher {
            inner: Inner::new(key, backend),
            buffer: [0; 32],
            buffer_usage: 0,
        }
    }

    pub fn backend(&self) -> Backend {
        match &self.inner {
            Inner::Portable(_) => Backend::Portable,
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(_) => Backend::Avx2,
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(_) => Backend::Sse41,
            #[cfg(all(target_arch = "aarch64", not(feature = "portable-only")))]
            Inner::Neon(_) => Backend::Neon,
        }
    }

    pub fn reset(&mut self, key: Key) {
        let backend = self.backend();
        self.inner = Inner::new(key, backend);
        self.buffer_usage = 0;
    }

    pub fn update(&mut self, mut bytes: &[u8]) {
        let capacity = 32 - self.buffer_usage;
        if bytes.len() < capacity {
            let n = bytes.len();
            self.buffer[self.buffer_usage..self.buffer_usage + n].copy_from_slice(bytes);
            self.buffer_usage += n;
            return;
        }
        if self.buffer_usage != 0 {
            let mut tmp = [0u8; 32];
            tmp[..self.buffer_usage].copy_from_slice(&self.buffer[..self.buffer_usage]);
            tmp[self.buffer_usage..32].copy_from_slice(&bytes[..capacity]);
            self.inner.update_packet(&tmp);
            bytes = &bytes[capacity..];
        }
        let (chunks, rest) = bytes.as_chunks::<32>();
        for chunk in chunks {
            self.inner.update_packet(chunk);
        }
        self.buffer_usage = rest.len();
        self.buffer[..rest.len()].copy_from_slice(rest);
    }

    pub fn finalize64(&self) -> u64 {
        let mut inner = self.inner.clone_inner();
        if self.buffer_usage != 0 {
            inner.update_remainder(&self.buffer[..self.buffer_usage]);
        }
        inner.finalize64()
    }

    pub fn finalize128(&self) -> [u64; 2] {
        let mut inner = self.inner.clone_inner();
        if self.buffer_usage != 0 {
            inner.update_remainder(&self.buffer[..self.buffer_usage]);
        }
        inner.finalize128()
    }

    pub fn finalize256(&self) -> [u64; 4] {
        let mut inner = self.inner.clone_inner();
        if self.buffer_usage != 0 {
            inner.update_remainder(&self.buffer[..self.buffer_usage]);
        }
        inner.finalize256()
    }
}

/// One-shot 64-bit HighwayHash using the best backend for the current CPU.
pub fn fairway64(key: Key, bytes: &[u8]) -> u64 {
    let mut h = HighwayHasher::new(key);
    h.update(bytes);
    h.finalize64()
}

/// One-shot 128-bit HighwayHash using the best backend for the current CPU.
pub fn fairway128(key: Key, bytes: &[u8]) -> [u64; 2] {
    let mut h = HighwayHasher::new(key);
    h.update(bytes);
    h.finalize128()
}

/// One-shot 256-bit HighwayHash using the best backend for the current CPU.
pub fn fairway256(key: Key, bytes: &[u8]) -> [u64; 4] {
    let mut h = HighwayHasher::new(key);
    h.update(bytes);
    h.finalize256()
}

/// Portable scalar implementation, always available regardless of CPU.
pub mod scalar {
    pub use crate::portable::{hash128, hash256, hash64, State};
}
