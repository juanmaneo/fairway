//! AVX2 SipTreeHash (port of the AVX2 `SipTreeHashT` in `sip_tree_hash.cc`).
//!
//! The four SipHash chains of the tree hash are the four 64-bit lanes of one
//! 256-bit register per state word, so a whole 32-byte packet is absorbed by
//! one vectorized SipHash round sequence. The result is identical to the
//! portable `tree_compute_scalar` in `sip.rs`.
//!
//! The entry point is `unsafe` because it executes AVX2 instructions; callers
//! must ensure the CPU supports AVX2.

use core::arch::x86_64::*;

use crate::sip::reduce;
use crate::{Key, SipHashKey};

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: AVX2 is required for `_mm256_xor_si256`; callers guarantee it.
    unsafe { _mm256_xor_si256(a, b) }
}

#[inline(always)]
unsafe fn add(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: AVX2 is required for `_mm256_add_epi64`; callers guarantee it.
    unsafe { _mm256_add_epi64(a, b) }
}

/// Rotates each 64-bit lane left by 13 bits.
#[inline(always)]
unsafe fn rotate_left13(v: __m256i) -> __m256i {
    // SAFETY: AVX2 is required for all `_mm256_*` intrinsics; callers guarantee it.
    unsafe { _mm256_or_si256(_mm256_slli_epi64::<13>(v), _mm256_srli_epi64::<51>(v)) }
}

/// Rotates each 64-bit lane left by 17 bits.
#[inline(always)]
unsafe fn rotate_left17(v: __m256i) -> __m256i {
    // SAFETY: AVX2 is required for all `_mm256_*` intrinsics; callers guarantee it.
    unsafe { _mm256_or_si256(_mm256_slli_epi64::<17>(v), _mm256_srli_epi64::<47>(v)) }
}

/// Rotates each 64-bit lane left by 21 bits.
#[inline(always)]
unsafe fn rotate_left21(v: __m256i) -> __m256i {
    // SAFETY: AVX2 is required for all `_mm256_*` intrinsics; callers guarantee it.
    unsafe { _mm256_or_si256(_mm256_slli_epi64::<21>(v), _mm256_srli_epi64::<43>(v)) }
}

/// Rotates each 64-bit lane left by 16 bits: a single byte shuffle.
#[inline(always)]
unsafe fn rotate_left16(v: __m256i) -> __m256i {
    // SAFETY: AVX2 is required for `_mm256_shuffle_epi8`; callers guarantee it.
    unsafe {
        // Output byte j takes input byte control[j] of the same 128-bit half
        // (`set_epi64x` lists the highest lane first).
        let control = _mm256_set_epi64x(
            0x0D0C0B0A09080F0E,
            0x0504030201000706,
            0x0D0C0B0A09080F0E,
            0x0504030201000706,
        );
        _mm256_shuffle_epi8(v, control)
    }
}

/// Rotates each 64-bit lane left by 32 bits: swaps its two 32-bit halves.
#[inline(always)]
unsafe fn rotate_left32(v: __m256i) -> __m256i {
    // SAFETY: AVX2 is required for `_mm256_shuffle_epi32`; callers guarantee it.
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
        // SAFETY: the caller guarantees AVX2 support; the unaligned load reads
        // exactly the 32 bytes of the key.
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
        // SAFETY: the caller guarantees AVX2 support.
        unsafe {
            self.v3 = xor(self.v3, packet);
            self.compress::<U>();
            self.v0 = xor(self.v0, packet);
        }
    }

    /// The four lane hashes.
    #[inline(always)]
    unsafe fn finalize(mut self) -> __m256i {
        // SAFETY: the caller guarantees AVX2 support.
        unsafe {
            // Mix in bits to avoid leaking the key if all packets were zero.
            self.v2 = xor(self.v2, _mm256_set1_epi64x(0xFF));
            self.compress::<F>();
            xor(xor(self.v0, self.v1), xor(self.v2, self.v3))
        }
    }

    #[inline(always)]
    unsafe fn compress<const ROUNDS: usize>(&mut self) {
        // SAFETY: the caller guarantees AVX2 support.
        unsafe {
            for _ in 0..ROUNDS {
                // ARX network: add, rotate, exclusive-or.
                self.v0 = add(self.v0, self.v1);
                self.v2 = add(self.v2, self.v3);
                self.v1 = rotate_left13(self.v1);
                self.v3 = rotate_left16(self.v3);
                self.v1 = xor(self.v1, self.v0);
                self.v3 = xor(self.v3, self.v2);

                self.v0 = rotate_left32(self.v0);

                self.v2 = add(self.v2, self.v1);
                self.v0 = add(self.v0, self.v3);
                self.v1 = rotate_left17(self.v1);
                self.v3 = rotate_left21(self.v3);
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
    // Whole 32-bit words are read with one masked load, which does not touch
    // (or fault on) the masked-off words past the end of the input. Copying
    // the tail into a buffer first would cost a store-forwarding stall.
    let whole_words = remainder >> 2; // 0..=7

    // One byte per word, 0xFF for each word to load (0 words => 0, 7 words =>
    // 0x00FF_FFFF_FFFF_FFFF), widened to a 32-bit mask below.
    let packed_mask = 0x00FF_FFFF_FFFF_FFFFu64 >> ((7 - whole_words) * 8);

    let mut packet4 = (remainder as u32) << 24;
    for (i, byte) in tail[whole_words * 4..].iter().enumerate() {
        packet4 += (*byte as u32) << (i * 8);
    }

    // SAFETY: the caller guarantees AVX2 support. Only the first
    // `whole_words` 32-bit words of `tail` are read, all within its bounds.
    unsafe {
        let mask = _mm256_cvtepi8_epi32(_mm_cvtsi64_si128(packed_mask as i64));
        let packet28 = _mm256_maskload_epi32(tail.as_ptr() as *const i32, mask);
        // The top word of `packet28` is zero; replace it with `packet4`.
        _mm256_blend_epi32::<0x80>(packet28, _mm256_set1_epi32(packet4 as i32))
    }
}

/// SipTreeHash with `U` update and `F` finalization rounds, equal to the
/// portable `tree_compute_scalar::<U, F>`.
///
/// # Safety
/// CPU must support AVX2.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn tree_hash<const U: usize, const F: usize>(key: Key, bytes: &[u8]) -> u64 {
    let (packets, tail) = bytes.as_chunks::<32>();
    let mut hashes = [0u64; 4];
    // SAFETY: the caller guarantees AVX2 support; the unaligned loads read
    // exactly the 32 bytes of each packet, and the store writes `hashes`.
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
    use crate::vectors::{SIP_TREE_HASH13_OUTPUT, SIP_TREE_HASH_OUTPUT, SIP_TREE_KEY};

    #[test]
    fn avx2_tree_hash_golden() {
        // As in the other backends' tests: the skip guard is compiled out
        // under `cfg(coverage)` (coverage machines have AVX2).
        #[cfg(not(coverage))]
        if !super::super::has_avx2() {
            return;
        }
        let input: Vec<u8> = (0..65536).map(|i| i as u8).collect();
        let key = Key(SIP_TREE_KEY);
        for (size, expected) in SIP_TREE_HASH_OUTPUT {
            // SAFETY: the test returned early unless the CPU has AVX2.
            let got = unsafe { tree_hash::<2, 4>(key, &input[..size]) };
            assert_eq!(got, expected, "avx2 tree24 at {size}");
        }
        for (size, expected) in SIP_TREE_HASH13_OUTPUT {
            // SAFETY: the test returned early unless the CPU has AVX2.
            let got = unsafe { tree_hash::<1, 3>(key, &input[..size]) };
            assert_eq!(got, expected, "avx2 tree13 at {size}");
        }
    }
}
