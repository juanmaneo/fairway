//! AVX-512VL SipTreeHash: `avx2_sip_tree.rs` re-encoded with EVEX
//! instructions on 256-bit registers.
//!
//! There is no upstream AVX-512 tree hash, so this is not a port of
//! `sip_tree_hash.cc`: the four SipHash chains are the four 64-bit lanes of
//! one 256-bit register per state word, exactly as in the AVX2 version, and
//! the result is identical to the portable `tree_compute_scalar` in `sip.rs`.
//! The substitutions against `avx2_sip_tree.rs`:
//!
//! * `_mm256_rol_epi64` replaces the shift-shift-or: one instruction instead
//!   of three, and one cycle shorter on the dependency chain for the
//!   rotations by 13, 17 and 21; for the rotation by 16 it only saves the
//!   shuffle-control constant. The rotation by 32 stays a
//!   `_mm256_shuffle_epi32`, which is just as short,
//! * `_mm256_maskz_loadu_epi8` replaces the mask construction and
//!   `_mm256_maskload_epi32` of the final packet.
//!
//! Whether this beats the AVX2 code has not been measured: on some CPUs
//! 256-bit AVX-512VL code runs at a lower clock than AVX2 (an earlier
//! measurement of the AVX-512VL HighwayHash found it slower than AVX2 on
//! small inputs). Compare `cargo run --release --example bench` with and
//! without `--features avx512`.
//!
//! The entry point is `unsafe` because it executes AVX-512 instructions;
//! callers must ensure the CPU supports avx512f + avx512vl + avx512bw (see
//! `has_avx512`).

// The AVX-512 intrinsics used below were stabilized in Rust 1.89; this module
// only compiles with the `avx512` feature, which therefore requires 1.89+.
#![allow(clippy::incompatible_msrv)]

use core::arch::x86_64::*;

use crate::sip::{final_packet_word, reduce};
use crate::{Key, SipHashKey};

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: AVX2 (implied by AVX-512F) is required for `_mm256_xor_si256`;
    // callers guarantee the AVX-512 features.
    unsafe { _mm256_xor_si256(a, b) }
}

#[inline(always)]
unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: AVX2 (implied by AVX-512F) is required for `_mm256_add_epi64`;
    // callers guarantee the AVX-512 features.
    unsafe { _mm256_add_epi64(a, b) }
}

/// Rotates each 64-bit lane left by `BITS` bits.
#[inline(always)]
unsafe fn rotate_left<const BITS: i32>(v: __m256i) -> __m256i {
    // SAFETY: avx512f + avx512vl are required for `_mm256_rol_epi64`; callers
    // guarantee the AVX-512 features.
    unsafe { _mm256_rol_epi64::<BITS>(v) }
}

/// Rotates each 64-bit lane left by 32 bits: swaps its two 32-bit halves.
#[inline(always)]
unsafe fn rotate_left32(v: __m256i) -> __m256i {
    // SAFETY: AVX2 (implied by AVX-512F) is required for
    // `_mm256_shuffle_epi32`; callers guarantee the AVX-512 features.
    unsafe { _mm256_shuffle_epi32(v, 0b10_11_00_01) }
}

/// The four SipHash states, one per lane.
struct State<const UPDATE_ROUNDS: usize, const FINALIZE_ROUNDS: usize> {
    v0: __m256i,
    v1: __m256i,
    v2: __m256i,
    v3: __m256i,
}

impl<const U: usize, const F: usize> State<U, F> {
    /// Lane `i` is the SipHash state keyed with `key[i] ^ (4 | i)`, as in the
    /// scalar tree hash.
    #[inline(always)]
    unsafe fn new(key: Key) -> Self {
        // SAFETY: the caller guarantees the AVX-512 features; the unaligned
        // load reads exactly the 32 bytes of the key.
        unsafe {
            let lane_tags = _mm256_set_epi64x(4 | 3, 4 | 2, 4 | 1, 4);
            let key = xor(
                _mm256_loadu_si256(key.0.as_ptr() as *const __m256i),
                lane_tags,
            );
            State {
                v0: xor(_mm256_set1_epi64x(0x736f6d6570736575), key),
                v1: xor(_mm256_set1_epi64x(0x646f72616e646f6d), key),
                v2: xor(_mm256_set1_epi64x(0x6c7967656e657261), key),
                v3: xor(_mm256_set1_epi64x(0x7465646279746573), key),
            }
        }
    }

    #[inline(always)]
    unsafe fn update(&mut self, packet: __m256i) {
        // SAFETY: the caller guarantees the AVX-512 features.
        unsafe {
            self.v3 = xor(self.v3, packet);
            self.compress::<U>();
            self.v0 = xor(self.v0, packet);
        }
    }

    /// The four lane hashes.
    #[inline(always)]
    unsafe fn finalize(mut self) -> __m256i {
        // SAFETY: the caller guarantees the AVX-512 features.
        unsafe {
            // Mix in bits to avoid leaking the key if all packets were zero.
            self.v2 = xor(self.v2, _mm256_set1_epi64x(0xFF));
            self.compress::<F>();
            xor(xor(self.v0, self.v1), xor(self.v2, self.v3))
        }
    }

    #[inline(always)]
    unsafe fn compress<const ROUNDS: usize>(&mut self) {
        // SAFETY: the caller guarantees the AVX-512 features.
        unsafe {
            for _ in 0..ROUNDS {
                // ARX network: add, rotate, exclusive-or.
                self.v0 = add(self.v0, self.v1);
                self.v2 = add(self.v2, self.v3);
                self.v1 = rotate_left::<13>(self.v1);
                self.v3 = rotate_left::<16>(self.v3);
                self.v1 = xor(self.v1, self.v0);
                self.v3 = xor(self.v3, self.v2);

                self.v0 = rotate_left32(self.v0);

                self.v2 = add(self.v2, self.v1);
                self.v0 = add(self.v0, self.v3);
                self.v1 = rotate_left::<17>(self.v1);
                self.v3 = rotate_left::<21>(self.v3);
                self.v1 = xor(self.v1, self.v2);
                self.v3 = xor(self.v3, self.v0);

                self.v2 = rotate_left32(self.v2);
            }
        }
    }
}

/// The final 32-byte packet: the 0..=31 remaining bytes, zero padding, and
/// the number of remaining bytes in the top byte (with the last `len % 4`
/// bytes packed below it), the same packet the scalar code builds.
#[inline(always)]
unsafe fn load_final_packet(tail: &[u8]) -> __m256i {
    let remainder = tail.len();
    debug_assert!(remainder < 32);
    // One mask bit per byte: the whole 32-bit words of the tail (at most 28
    // bytes, so the shift cannot overflow).
    let mask: __mmask32 = (1u32 << (remainder & !3)) - 1;
    let packet4 = final_packet_word(tail);

    // SAFETY: the caller guarantees the AVX-512 features. The masked load
    // reads only the bytes selected by `mask`, all within `tail`, and does
    // not fault on the masked-off bytes.
    unsafe {
        let packet28 = _mm256_maskz_loadu_epi8(mask, tail.as_ptr() as *const i8);
        // The top word of `packet28` is zero; replace it with `packet4`.
        _mm256_blend_epi32::<0x80>(packet28, _mm256_set1_epi32(packet4 as i32))
    }
}

/// SipTreeHash with `U` update and `F` finalization rounds, equal to the
/// portable `tree_compute_scalar::<U, F>`.
///
/// # Safety
/// CPU must support avx512f + avx512vl + avx512bw.
#[target_feature(enable = "avx512f,avx512vl,avx512bw")]
pub(crate) unsafe fn tree_hash<const U: usize, const F: usize>(key: Key, bytes: &[u8]) -> u64 {
    let (packets, tail) = bytes.as_chunks::<32>();
    let mut hashes = [0u64; 4];
    // SAFETY: the caller guarantees the AVX-512 features; the unaligned loads
    // read exactly the 32 bytes of each packet, and the store writes
    // `hashes`.
    unsafe {
        let mut state = State::<U, F>::new(key);
        for packet in packets {
            state.update(_mm256_loadu_si256(packet.as_ptr() as *const __m256i));
        }
        state.update(load_final_packet(tail));
        _mm256_storeu_si256(hashes.as_mut_ptr() as *mut __m256i, state.finalize());
    }
    // The four lane hashes are reduced to one with a scalar SipHash.
    reduce::<U, F>(SipHashKey([key.0[0], key.0[1]]), &hashes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sip::{scalar_sip_tree_hash, scalar_sip_tree_hash13};
    use crate::vectors::{SIP_TREE_HASH13_OUTPUT, SIP_TREE_HASH_OUTPUT, SIP_TREE_KEY};

    /// Every length 0..=160 (the final packet's 32 tail lengths, with up to
    /// five whole packets before it) against the portable code.
    #[test]
    fn avx512_tree_hash_matches_scalar() {
        // Like the other AVX-512 tests: the skip guard is compiled out under
        // `cfg(coverage)`, which only passes on machines that have AVX-512.
        #[cfg(not(coverage))]
        if !super::super::has_avx512() {
            return;
        }
        let input: Vec<u8> = (0..160u32).map(|i| (i * 7 + 3) as u8).collect();
        let key = Key(SIP_TREE_KEY);
        for len in 0..=input.len() {
            let bytes = &input[..len];
            // SAFETY: the test returned early unless the CPU has AVX-512.
            unsafe {
                assert_eq!(
                    tree_hash::<2, 4>(key, bytes),
                    scalar_sip_tree_hash(key, bytes),
                    "avx512 tree24 at {len}"
                );
                assert_eq!(
                    tree_hash::<1, 3>(key, bytes),
                    scalar_sip_tree_hash13(key, bytes),
                    "avx512 tree13 at {len}"
                );
            }
        }
    }

    #[test]
    fn avx512_tree_hash_golden() {
        // As in `avx512_tree_hash_matches_scalar`.
        #[cfg(not(coverage))]
        if !super::super::has_avx512() {
            return;
        }
        let input: Vec<u8> = (0..65536).map(|i| i as u8).collect();
        let key = Key(SIP_TREE_KEY);
        for (size, expected) in SIP_TREE_HASH_OUTPUT {
            // SAFETY: the test returned early unless the CPU has AVX-512.
            let got = unsafe { tree_hash::<2, 4>(key, &input[..size]) };
            assert_eq!(got, expected, "avx512 tree24 at {size}");
        }
        for (size, expected) in SIP_TREE_HASH13_OUTPUT {
            // SAFETY: the test returned early unless the CPU has AVX-512.
            let got = unsafe { tree_hash::<1, 3>(key, &input[..size]) };
            assert_eq!(got, expected, "avx512 tree13 at {size}");
        }
    }
}
