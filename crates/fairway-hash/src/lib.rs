//! Rust port of Google's [HighwayHash](https://github.com/google/highwayhash)
//! and its companion SipHash implementation.
//!
//! HighwayHash is a strong pseudorandom function with security claims
//! (<https://arxiv.org/abs/1612.06257>), about 4x faster than SipHash and 10x
//! faster than BLAKE2. This crate reproduces the exact, frozen output of the
//! upstream C++ library (verified against its golden test vectors) and picks
//! the best available implementation at runtime:
//!
//! * AVX-512VL on x86_64 behind the off-by-default `avx512` cargo feature
//!   (requires Rust 1.89+ when enabled; the MSRV stays 1.88 without it),
//! * AVX2 and SSE4.1 on x86_64 (via `core::arch`, detected with `std::arch`),
//! * NEON on little-endian aarch64,
//! * a portable scalar fallback everywhere else (including big-endian
//!   targets, aarch64 among them: the NEON code loads native-endian words).
//!
//! # Examples
//!
//! ```
//! use fairway_hash::{HighwayHasher, Key};
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
//! use fairway_hash::{fairway64, Key};
//! assert_eq!(fairway64(Key([1, 2, 3, 4]), b"abc"), fairway64(Key([1, 2, 3, 4]), b"abc"));
//! ```
//!
//! When many small inputs share one key, a hasher built once does the key
//! schedule once; [`HighwayHasher::hash64`] then hashes each input from that
//! state (about 1.5x faster than [`fairway64`] for inputs under 1 KiB):
//!
//! ```
//! use fairway_hash::{fairway64, HighwayHasher, Key};
//! let key = Key([1, 2, 3, 4]);
//! let hasher = HighwayHasher::new(key);
//! assert_eq!(hasher.hash64(b"packet"), fairway64(key, b"packet"));
//! ```
//!
//! Many independent inputs are faster hashed together than one by one (see
//! [`fairway64_many`], and [`fairway64_many_par`] with the `std` feature to
//! add threads):
//!
//! ```
//! use fairway_hash::{fairway64, fairway64_many, Key};
//! let key = Key([1, 2, 3, 4]);
//! let inputs = [&b"one"[..], b"two", b"three"];
//! let mut out = [0u64; 3];
//! fairway64_many(key, &inputs, &mut out);
//! assert_eq!(out[2], fairway64(key, b"three"));
//! ```
//!
//! The SipHash family is included too: [`sip_hash`] and [`sip_hash13`], and
//! the 256-bit-keyed tree variants [`sip_tree_hash`] and [`sip_tree_hash13`]
//! (four interleaved SipHash chains over 32-byte packets, reduced to one
//! hash, on AVX-512VL (with the `avx512` feature), AVX2 or NEON where
//! available and portable code elsewhere;
//! [`reduce_sip_tree_hash`] combines hashes of independent chunks):
//!
//! ```
//! use fairway_hash::{reduce_sip_tree_hash, sip_tree_hash, Key, SipHashKey};
//! let key = Key([1, 2, 3, 4]);
//! let hash = sip_tree_hash(key, b"some attacker-controlled input");
//! let combined = reduce_sip_tree_hash(SipHashKey([1, 2]), &[hash, hash ^ 1]);
//! assert_ne!(hash, combined);
//! ```

// Unit tests use `std` (the test harness needs it anyway), even without the
// `std` feature.
#![cfg_attr(all(not(feature = "std"), not(test)), no_std)]
// SIMD backends use `core::arch` intrinsics, which are `unsafe`; the unsafe
// code is confined to the `arch` module and only invoked after runtime
// CPU feature detection (or explicit `target_feature` callers).

#[cfg(all(
    any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little")
    ),
    not(feature = "portable-only")
))]
mod arch;
mod batch;
mod portable;
mod sip;
#[doc(hidden)]
#[rustfmt::skip] // Keep the golden tables compact, as transcribed.
pub mod vectors;

#[cfg(all(
    target_arch = "x86_64",
    feature = "avx512",
    not(feature = "portable-only")
))]
pub use arch::x86::avx512;
#[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
pub use arch::x86::{avx2, sse41};

pub use batch::{fairway128_many, fairway256_many, fairway64_many};
#[cfg(feature = "std")]
pub use batch::{fairway128_many_par, fairway256_many_par, fairway64_many_par};
pub use sip::{
    reduce_sip_tree_hash, reduce_sip_tree_hash13, sip_hash, sip_hash13, sip_tree_hash,
    sip_tree_hash13, SipHash13State, SipHashKey, SipHashState, SipState,
};

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
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    Avx512,
    Avx2,
    Sse41,
    Neon,
    Portable,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            Backend::Avx512 => "AVX512VL",
            Backend::Avx2 => "AVX2",
            Backend::Sse41 => "SSE4.1",
            Backend::Neon => "NEON",
            Backend::Portable => "Portable",
        }
    }
}

/// Pure x86_64 backend-selection logic, kept separate from CPU detection so
/// every branch is unit-testable on any machine (see `backend_tests`);
/// `best_backend` only adds the platform rules on top.
#[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
fn best_backend_for(avx512: bool, avx2: bool, sse41: bool) -> Backend {
    #[cfg(feature = "avx512")]
    if avx512 {
        return Backend::Avx512;
    }
    #[cfg(not(feature = "avx512"))]
    let _ = avx512; // No AVX-512 backend in this build.
    if avx2 {
        Backend::Avx2
    } else if sse41 {
        Backend::Sse41
    } else {
        Backend::Portable
    }
}

/// Best backend available on the current CPU (runtime-detected on x86_64).
pub fn best_backend() -> Backend {
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
    {
        #[cfg(feature = "avx512")]
        let avx512 = arch::x86::has_avx512();
        #[cfg(not(feature = "avx512"))]
        let avx512 = false;
        best_backend_for(avx512, arch::x86::has_avx2(), arch::x86::has_sse41())
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        not(feature = "portable-only")
    ))]
    {
        Backend::Neon
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", not(feature = "portable-only")),
        all(
            target_arch = "aarch64",
            target_endian = "little",
            not(feature = "portable-only")
        ),
    )))]
    {
        Backend::Portable
    }
}

#[derive(Clone)]
enum Inner {
    Portable(portable::State),
    #[cfg(all(
        target_arch = "x86_64",
        feature = "avx512",
        not(feature = "portable-only")
    ))]
    Avx512(arch::x86::avx512::State),
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
    Avx2(arch::x86::avx2::State),
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
    Sse41(arch::x86::sse41::State),
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        not(feature = "portable-only")
    ))]
    Neon(arch::aarch64::neon::State),
}

impl Inner {
    fn new(key: Key, backend: Backend) -> Self {
        match backend {
            #[cfg(all(
                target_arch = "x86_64",
                feature = "avx512",
                not(feature = "portable-only")
            ))]
            Backend::Avx512 if arch::x86::has_avx512() => {
                // SAFETY: AVX-512 support was just detected.
                Inner::Avx512(unsafe { arch::x86::entry::avx512::new(key) })
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Backend::Avx2 if arch::x86::has_avx2() => {
                // SAFETY: AVX2 support was just detected.
                Inner::Avx2(unsafe { arch::x86::entry::avx2::new(key) })
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Backend::Sse41 if arch::x86::has_sse41() => {
                // SAFETY: SSE4.1 support was just detected.
                Inner::Sse41(unsafe { arch::x86::entry::sse41::new(key) })
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                not(feature = "portable-only")
            ))]
            Backend::Neon => Inner::Neon(arch::aarch64::neon::State::new(key)),
            // Portable, or a backend this CPU/build cannot run.
            _ => Inner::Portable(portable::State::new(key)),
        }
    }

    /// Feeds whole packets, dispatching on the backend once per call rather
    /// than once per packet.
    fn update_packets(&mut self, packets: &[[u8; 32]]) {
        match self {
            Inner::Portable(s) => {
                for packet in packets {
                    s.update_packet(packet);
                }
            }
            #[cfg(all(
                target_arch = "x86_64",
                feature = "avx512",
                not(feature = "portable-only")
            ))]
            Inner::Avx512(s) => {
                // SAFETY: an `Inner::Avx512` exists only when AVX-512 was detected.
                unsafe { arch::x86::entry::avx512::update_packets(s, packets) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => {
                // SAFETY: an `Inner::Avx2` exists only when AVX2 was detected.
                unsafe { arch::x86::entry::avx2::update_packets(s, packets) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => {
                // SAFETY: an `Inner::Sse41` exists only when SSE4.1 was detected.
                unsafe { arch::x86::entry::sse41::update_packets(s, packets) }
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                not(feature = "portable-only")
            ))]
            Inner::Neon(s) => {
                for packet in packets {
                    s.update_packet(packet);
                }
            }
        }
    }

    /// Finalizes a copy of the state after absorbing `remainder` (0..=31
    /// bytes), leaving the hasher itself untouched.
    fn finalize64(&self, remainder: &[u8]) -> u64 {
        match self {
            Inner::Portable(s) => {
                let mut s = s.clone();
                if !remainder.is_empty() {
                    s.update_remainder(remainder);
                }
                s.finalize64()
            }
            #[cfg(all(
                target_arch = "x86_64",
                feature = "avx512",
                not(feature = "portable-only")
            ))]
            Inner::Avx512(s) => {
                // SAFETY: an `Inner::Avx512` exists only when AVX-512 was detected.
                unsafe { arch::x86::entry::avx512::finalize64(s, remainder) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => {
                // SAFETY: an `Inner::Avx2` exists only when AVX2 was detected.
                unsafe { arch::x86::entry::avx2::finalize64(s, remainder) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => {
                // SAFETY: an `Inner::Sse41` exists only when SSE4.1 was detected.
                unsafe { arch::x86::entry::sse41::finalize64(s, remainder) }
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                not(feature = "portable-only")
            ))]
            Inner::Neon(s) => {
                let mut s = *s;
                if !remainder.is_empty() {
                    s.update_remainder(remainder);
                }
                s.finalize64()
            }
        }
    }

    /// Finalizes a copy of the state after absorbing `remainder` (0..=31
    /// bytes), leaving the hasher itself untouched.
    fn finalize128(&self, remainder: &[u8]) -> [u64; 2] {
        match self {
            Inner::Portable(s) => {
                let mut s = s.clone();
                if !remainder.is_empty() {
                    s.update_remainder(remainder);
                }
                s.finalize128()
            }
            #[cfg(all(
                target_arch = "x86_64",
                feature = "avx512",
                not(feature = "portable-only")
            ))]
            Inner::Avx512(s) => {
                // SAFETY: an `Inner::Avx512` exists only when AVX-512 was detected.
                unsafe { arch::x86::entry::avx512::finalize128(s, remainder) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => {
                // SAFETY: an `Inner::Avx2` exists only when AVX2 was detected.
                unsafe { arch::x86::entry::avx2::finalize128(s, remainder) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => {
                // SAFETY: an `Inner::Sse41` exists only when SSE4.1 was detected.
                unsafe { arch::x86::entry::sse41::finalize128(s, remainder) }
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                not(feature = "portable-only")
            ))]
            Inner::Neon(s) => {
                let mut s = *s;
                if !remainder.is_empty() {
                    s.update_remainder(remainder);
                }
                s.finalize128()
            }
        }
    }

    /// Finalizes a copy of the state after absorbing `remainder` (0..=31
    /// bytes), leaving the hasher itself untouched.
    fn finalize256(&self, remainder: &[u8]) -> [u64; 4] {
        match self {
            Inner::Portable(s) => {
                let mut s = s.clone();
                if !remainder.is_empty() {
                    s.update_remainder(remainder);
                }
                s.finalize256()
            }
            #[cfg(all(
                target_arch = "x86_64",
                feature = "avx512",
                not(feature = "portable-only")
            ))]
            Inner::Avx512(s) => {
                // SAFETY: an `Inner::Avx512` exists only when AVX-512 was detected.
                unsafe { arch::x86::entry::avx512::finalize256(s, remainder) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => {
                // SAFETY: an `Inner::Avx2` exists only when AVX2 was detected.
                unsafe { arch::x86::entry::avx2::finalize256(s, remainder) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => {
                // SAFETY: an `Inner::Sse41` exists only when SSE4.1 was detected.
                unsafe { arch::x86::entry::sse41::finalize256(s, remainder) }
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                not(feature = "portable-only")
            ))]
            Inner::Neon(s) => {
                let mut s = *s;
                if !remainder.is_empty() {
                    s.update_remainder(remainder);
                }
                s.finalize256()
            }
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
            #[cfg(all(
                target_arch = "x86_64",
                feature = "avx512",
                not(feature = "portable-only")
            ))]
            Inner::Avx512(_) => Backend::Avx512,
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(_) => Backend::Avx2,
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(_) => Backend::Sse41,
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                not(feature = "portable-only")
            ))]
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
            self.buffer[self.buffer_usage..].copy_from_slice(&bytes[..capacity]);
            self.inner
                .update_packets(core::slice::from_ref(&self.buffer));
            bytes = &bytes[capacity..];
        }
        let (chunks, rest) = bytes.as_chunks::<32>();
        self.inner.update_packets(chunks);
        self.buffer_usage = rest.len();
        self.buffer[..rest.len()].copy_from_slice(rest);
    }

    pub fn finalize64(&self) -> u64 {
        self.inner.finalize64(&self.buffer[..self.buffer_usage])
    }

    pub fn finalize128(&self) -> [u64; 2] {
        self.inner.finalize128(&self.buffer[..self.buffer_usage])
    }

    pub fn finalize256(&self) -> [u64; 4] {
        self.inner.finalize256(&self.buffer[..self.buffer_usage])
    }

    /// The hash of everything fed so far followed by `bytes`, without
    /// changing the hasher. On a hasher that has absorbed nothing this is
    /// [`fairway64`] with the key schedule and CPU detection already done,
    /// which matters when hashing many small inputs with one key:
    ///
    /// ```
    /// use fairway_hash::{fairway64, HighwayHasher, Key};
    /// let key = Key([1, 2, 3, 4]);
    /// let hasher = HighwayHasher::new(key);
    /// for packet in [&b"abc"[..], b"defg"] {
    ///     assert_eq!(hasher.hash64(packet), fairway64(key, packet));
    /// }
    /// ```
    pub fn hash64(&self, bytes: &[u8]) -> u64 {
        self.hash(bytes, Inner::finalize64, HighwayHasher::finalize64)
    }

    /// See [`HighwayHasher::hash64`].
    pub fn hash128(&self, bytes: &[u8]) -> [u64; 2] {
        self.hash(bytes, Inner::finalize128, HighwayHasher::finalize128)
    }

    /// See [`HighwayHasher::hash64`].
    pub fn hash256(&self, bytes: &[u8]) -> [u64; 4] {
        self.hash(bytes, Inner::finalize256, HighwayHasher::finalize256)
    }

    fn hash<T>(
        &self,
        bytes: &[u8],
        finish: fn(&Inner, &[u8]) -> T,
        finalize: fn(&HighwayHasher) -> T,
    ) -> T {
        if self.buffer_usage == 0 {
            return hash_from(self.inner.clone(), bytes, finish);
        }
        let mut hasher = HighwayHasher {
            inner: self.inner.clone(),
            buffer: self.buffer,
            buffer_usage: self.buffer_usage,
        };
        hasher.update(bytes);
        finalize(&hasher)
    }
}

/// Hashes `bytes` starting from `inner`: whole packets straight from the
/// input, the tail straight into finalization, without the incremental
/// hasher's buffer.
fn hash_from<T>(mut inner: Inner, bytes: &[u8], finish: fn(&Inner, &[u8]) -> T) -> T {
    let (packets, remainder) = bytes.as_chunks::<32>();
    inner.update_packets(packets);
    finish(&inner, remainder)
}

/// Hashes `bytes` in one go from a fresh state.
fn one_shot<T>(key: Key, bytes: &[u8], finish: fn(&Inner, &[u8]) -> T) -> T {
    hash_from(Inner::new(key, best_backend()), bytes, finish)
}

/// One-shot 64-bit HighwayHash using the best backend for the current CPU.
pub fn fairway64(key: Key, bytes: &[u8]) -> u64 {
    one_shot(key, bytes, Inner::finalize64)
}

/// One-shot 128-bit HighwayHash using the best backend for the current CPU.
pub fn fairway128(key: Key, bytes: &[u8]) -> [u64; 2] {
    one_shot(key, bytes, Inner::finalize128)
}

/// One-shot 256-bit HighwayHash using the best backend for the current CPU.
pub fn fairway256(key: Key, bytes: &[u8]) -> [u64; 4] {
    one_shot(key, bytes, Inner::finalize256)
}

/// Portable scalar implementation, always available regardless of CPU.
pub mod scalar {
    pub use crate::portable::{hash128, hash256, hash64, State};
    pub use crate::sip::{
        scalar_sip_tree_hash as sip_tree_hash, scalar_sip_tree_hash13 as sip_tree_hash13,
    };
}

#[cfg(all(test, target_arch = "x86_64", not(feature = "portable-only")))]
mod backend_tests {
    use super::Backend;

    // Exercising the dispatch logic with synthetic inputs keeps every region
    // covered even on AVX2 machines, where only the `Avx2` arm is live.

    #[test]
    fn avx2_wins_over_sse41() {
        assert_eq!(super::best_backend_for(false, true, true), Backend::Avx2);
        assert_eq!(super::best_backend_for(false, true, false), Backend::Avx2);
    }

    #[test]
    fn sse41_used_when_avx2_is_missing() {
        assert_eq!(super::best_backend_for(false, false, true), Backend::Sse41);
    }

    #[test]
    fn portable_used_when_no_simd() {
        assert_eq!(
            super::best_backend_for(false, false, false),
            Backend::Portable
        );
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn avx512_wins_when_present() {
        assert_eq!(super::best_backend_for(true, true, true), Backend::Avx512);
        assert_eq!(super::best_backend_for(true, false, false), Backend::Avx512);
    }

    #[cfg(not(feature = "avx512"))]
    #[test]
    fn avx512_flag_is_ignored_when_compiled_out() {
        assert_eq!(super::best_backend_for(true, true, false), Backend::Avx2);
        assert_eq!(
            super::best_backend_for(true, false, false),
            Backend::Portable
        );
    }
}
