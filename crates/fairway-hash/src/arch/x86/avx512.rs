//! AVX-512VL HighwayHash state: the AVX2 algorithm re-encoded with EVEX
//! instructions on 256-bit registers.
//!
//! There is no upstream AVX-512 implementation, so this is not a port of a
//! `hh_*.h` file: it is `avx2.rs` with shorter dependency chains, and the
//! golden tests below prove the output is identical on every input size
//! 0..=64. The substitutions against `avx2.rs`:
//!
//! * `_mm256_rolv_epi32` replaces the three-instruction `rotate32_by`,
//! * `_mm256_ternarylogic_epi32` fuses the four XORs of
//!   `xor_by_shift128_left12` into two,
//! * `_mm256_maskz_loadu_epi8` replaces the `int_mask` + `_mm_maskload_epi32`
//!   pair in `update_remainder`.
//!
//! All methods are `unsafe` because they execute AVX-512 instructions;
//! callers must ensure the CPU supports avx512f + avx512vl + avx512bw (e.g.
//! `is_x86_feature_detected!("avx512f")` and friends).

// The AVX-512 intrinsics used below were stabilized in Rust 1.89; this module
// only compiles with the `avx512` feature, which therefore requires 1.89+.
#![allow(clippy::incompatible_msrv)]

use core::arch::x86_64::*;

use crate::Key;

// Lane order matches `_mm256_set_epi64x(p3, p2, p1, p0)`: index 0 is the
// most-significant lane. The C++ `V4x64U(a, b, c, d)` constructor is
// `set_epi64x(d, c, b, a)`, so array element 0 corresponds to `a`.
const INIT0: [u64; 4] = [
    0xdbe6d5d5fe4cce2f,
    0xa4093822299f31d0,
    0x13198a2e03707344,
    0x243f6a8885a308d3,
];
const INIT1: [u64; 4] = [
    0x3bd39e10cb0ef593,
    0xc0acf169b5f18a8c,
    0xbe5466cf34e90c6c,
    0x452821e638d01377,
];

#[inline(always)]
unsafe fn rotate64_by32(v: __m256i) -> __m256i {
    // SAFETY: AVX2 (implied by AVX-512F) is required for `_mm256_shuffle_epi32`;
    // callers guarantee the AVX-512 features.
    unsafe { _mm256_shuffle_epi32(v, 0b10_11_00_01) }
}

#[inline(always)]
unsafe fn rotate32_by(v: __m256i, count: __m256i) -> __m256i {
    // SAFETY: avx512f + avx512vl are required for `_mm256_rolv_epi32`;
    // callers guarantee the AVX-512 features.
    unsafe { _mm256_rolv_epi32(v, count) }
}

#[inline(always)]
unsafe fn permute(v: __m256i) -> __m256i {
    // SAFETY: AVX2 (implied by AVX-512F) is required for
    // `_mm256_permutevar8x32_epi32`; callers guarantee the AVX-512 features.
    unsafe {
        // 32-bit lane indices [5, 4, 7, 6, 1, 0, 3, 2] (out lane j takes v lane
        // idx[j]); equivalent to the portable Permute [rot32(v[2]), rot32(v[3]),
        // rot32(v[0]), rot32(v[1])].
        let idx = _mm256_set_epi32(2, 3, 0, 1, 6, 7, 4, 5);
        _mm256_permutevar8x32_epi32(v, idx)
    }
}

#[inline(always)]
unsafe fn zipper_merge(v: __m256i) -> __m256i {
    // SAFETY: AVX2 (implied by AVX-512F) is required for `_mm256_shuffle_epi8`;
    // callers guarantee the AVX-512 features.
    unsafe {
        // _mm256_shuffle_epi8 control words per 128-bit lane; lane0 ctrl is
        // the low 8 bytes of the low 64-bit word of each 128-bit half.
        // Desired byte permutation (output byte j <- input byte perm[j]):
        //   lane0: 3 12 2 5 14 1 15 0, lane1: 11 4 10 13 9 6 8 7
        let lane0 = 0x000F010E05020C03u64;
        let lane1 = 0x070806090D0A040Bu64;
        let ctrl = _mm256_set_epi64x(lane1 as i64, lane0 as i64, lane1 as i64, lane0 as i64);
        _mm256_shuffle_epi8(v, ctrl)
    }
}

/// XORs x << 1 and x << 2 into `out` after clearing the upper two bits of x
/// (Lemire's reduction, see `XorByShift128Left12`).
#[inline(always)]
unsafe fn xor_by_shift128_left12(ba: __m256i, out: &mut __m256i) {
    // SAFETY: the AVX-512 features are required for `_mm256_ternarylogic_epi32`;
    // callers guarantee them.
    unsafe {
        let zero = _mm256_xor_si256(ba, ba);
        let top_bits2 = _mm256_srli_epi64(ba, 64 - 2);
        let shifted1_unmasked = _mm256_add_epi64(ba, ba);
        let top_bits1 = _mm256_srli_epi64(ba, 64 - 1);
        let ones = _mm256_cmpeq_epi64(ba, ba);
        let upper_8bytes = _mm256_slli_si256(ones, 8);
        let shifted2 = _mm256_add_epi64(shifted1_unmasked, shifted1_unmasked);
        let upper_bit_of_128 = _mm256_slli_epi64(upper_8bytes, 63);
        let new_low_bits2 = _mm256_unpacklo_epi64(zero, top_bits2);
        let shifted1 = _mm256_andnot_si256(upper_bit_of_128, shifted1_unmasked);
        let new_low_bits1 = _mm256_unpacklo_epi64(zero, top_bits1);
        // 0x96 is a ^ b ^ c per bit, so each ternarylogic fuses one XOR of
        // three operands into a single instruction.
        let t = _mm256_ternarylogic_epi32(*out, shifted2, new_low_bits2, 0x96);
        *out = _mm256_ternarylogic_epi32(t, shifted1, new_low_bits1, 0x96);
    }
}

/// Modular reduction by x^128 + x^2 + x for two interleaved 256-bit numbers.
#[inline(always)]
unsafe fn modular_reduction(b32a32: __m256i, b10a10: __m256i) -> __m256i {
    let mut out = b10a10;
    // SAFETY: the caller of this function guarantees the AVX-512 features.
    unsafe { xor_by_shift128_left12(b32a32, &mut out) };
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectors::{EXPECTED128, EXPECTED256, EXPECTED64, KEY64};

    fn inputs() -> Vec<u8> {
        // Empty string, then 00, 00 01, ... like the C++ test.
        (0..=64).collect()
    }

    #[test]
    fn avx512_golden64() {
        // Like the AVX2 tests: the coverage build (cargo-llvm-cov sets
        // `cfg(coverage)`) runs the AVX-512 code only on machines that have
        // it, so the skip guard is compiled out there; the unexecuted
        // `return` would otherwise count as an uncovered line.
        #[cfg(not(coverage))]
        if !super::super::has_avx512() {
            return;
        }
        let in_buf = inputs();
        let key = Key(KEY64);
        for size in 0..=64usize {
            // SAFETY: the test returned early unless the CPU has AVX-512.
            let mut s = unsafe { State::new(key) };
            let (chunks, rest) = in_buf[..size].as_chunks::<32>();
            for c in chunks {
                // SAFETY: AVX-512 detected above.
                unsafe { s.update_packet(c) };
            }
            if !rest.is_empty() {
                // SAFETY: AVX-512 detected above; `rest` is 1..=31 bytes.
                unsafe { s.update_remainder(rest) };
            }
            // SAFETY: AVX-512 detected above.
            assert_eq!(
                // SAFETY: AVX-512 detected above.
                unsafe { s.finalize64() },
                EXPECTED64[size],
                "avx512 64-bit at {size}"
            );
        }
    }

    #[test]
    fn avx512_golden128_256() {
        // As in `avx512_golden64`: the skip guard is compiled out under
        // `cfg(coverage)` on machines that have AVX-512.
        #[cfg(not(coverage))]
        if !super::super::has_avx512() {
            return;
        }
        let in_buf = inputs();
        let key = Key(KEY64);
        for size in 0..=64usize {
            // SAFETY: the test returned early unless the CPU has AVX-512.
            let mut s = unsafe { State::new(key) };
            let (chunks, rest) = in_buf[..size].as_chunks::<32>();
            for c in chunks {
                // SAFETY: AVX-512 detected above.
                unsafe { s.update_packet(c) };
            }
            if !rest.is_empty() {
                // SAFETY: AVX-512 detected above; `rest` is 1..=31 bytes.
                unsafe { s.update_remainder(rest) };
            }
            // SAFETY: AVX-512 detected above.
            assert_eq!(
                // SAFETY: AVX-512 detected above.
                unsafe { s.finalize128() },
                EXPECTED128[size],
                "avx512 128-bit at {size}"
            );
            // SAFETY: AVX-512 detected above.
            let mut s = unsafe { State::new(key) };
            let (chunks, rest) = in_buf[..size].as_chunks::<32>();
            for c in chunks {
                // SAFETY: AVX-512 detected above.
                unsafe { s.update_packet(c) };
            }
            if !rest.is_empty() {
                // SAFETY: AVX-512 detected above; `rest` is 1..=31 bytes.
                unsafe { s.update_remainder(rest) };
            }
            // SAFETY: AVX-512 detected above.
            assert_eq!(
                // SAFETY: AVX-512 detected above.
                unsafe { s.finalize256() },
                EXPECTED256[size],
                "avx512 256-bit at {size}"
            );
        }
    }
}

/// AVX-512VL hash state. `Copy` because it is just four 256-bit registers.
#[derive(Clone, Copy)]
pub struct State {
    v0: __m256i,
    v1: __m256i,
    mul0: __m256i,
    mul1: __m256i,
}

impl State {
    /// # Safety
    /// CPU must support avx512f + avx512vl + avx512bw.
    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    pub unsafe fn new(key: Key) -> Self {
        // SAFETY: the caller guarantees the AVX-512 features; the unaligned
        // loads read only the 32 bytes of the key/array constants.
        unsafe {
            let init0 = _mm256_loadu_si256(INIT0.as_ptr() as *const __m256i);
            let init1 = _mm256_loadu_si256(INIT1.as_ptr() as *const __m256i);
            let key_v = _mm256_loadu_si256(key.0.as_ptr() as *const __m256i);
            State {
                v0: _mm256_xor_si256(key_v, init0),
                v1: _mm256_xor_si256(rotate64_by32(key_v), init1),
                mul0: init0,
                mul1: init1,
            }
        }
    }

    /// # Safety
    /// CPU must support avx512f + avx512vl + avx512bw.
    #[inline]
    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    pub unsafe fn update_packet(&mut self, packet: &[u8; 32]) {
        // SAFETY: the caller guarantees the AVX-512 features; the unaligned
        // load reads exactly the 32 bytes of `packet`.
        unsafe {
            let p = _mm256_loadu_si256(packet.as_ptr() as *const __m256i);
            self.update(p);
        }
    }

    #[inline(always)]
    unsafe fn update(&mut self, packet: __m256i) {
        // SAFETY: the caller guarantees the AVX-512 features.
        unsafe {
            self.v1 = _mm256_add_epi64(self.v1, packet);
            self.v1 = _mm256_add_epi64(self.v1, self.mul0);
            self.mul0 = _mm256_xor_si256(
                self.mul0,
                _mm256_mul_epu32(self.v1, _mm256_srli_epi64(self.v0, 32)),
            );
            self.v0 = _mm256_add_epi64(self.v0, self.mul1);
            self.mul1 = _mm256_xor_si256(
                self.mul1,
                _mm256_mul_epu32(self.v0, _mm256_srli_epi64(self.v1, 32)),
            );
            self.v0 = _mm256_add_epi64(self.v0, zipper_merge(self.v1));
            self.v1 = _mm256_add_epi64(self.v1, zipper_merge(self.v0));
        }
    }

    /// # Safety
    /// CPU must support avx512f + avx512vl + avx512bw. `bytes.len()` must be
    /// in 1..=31.
    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    pub unsafe fn update_remainder(&mut self, bytes: &[u8]) {
        let size_mod32 = bytes.len();
        debug_assert!((1..32).contains(&size_mod32));
        // SAFETY: the caller guarantees the AVX-512 features and
        // `bytes.len()` in 1..=31; the masked loads read only the bytes
        // covered by their mask (masked-off lanes are not read), and the
        // `load3` overlap is in bounds.
        unsafe {
            let size256 = _mm256_set1_epi32(size_mod32 as i32);
            self.v0 = _mm256_add_epi64(self.v0, _mm256_shuffle_epi32(size256, 0b00_00_00_00));
            self.v1 = rotate32_by(self.v1, size256);

            let size_mod4 = size_mod32 & 3;
            let aligned = size_mod32 & !3;
            let remainder = bytes.as_ptr().add(aligned);

            if size_mod32 & 16 != 0 {
                let packet_l = _mm_loadu_si128(bytes.as_ptr() as *const __m128i);
                // Bytes 16..aligned land at their natural packet offset.
                let tail_bytes = aligned - 16;
                let mask = (1u16 << tail_bytes).wrapping_sub(1);
                let int_lanes =
                    _mm_maskz_loadu_epi8(mask as __mmask16, bytes.as_ptr().add(16) as *const i8);
                let last4 = super::load3_allow_read_before_and_return(remainder, size_mod4);
                let packet_h = _mm_insert_epi32(int_lanes, last4 as i32, 3);
                let packet = _mm256_inserti128_si256(_mm256_castsi128_si256(packet_l), packet_h, 1);
                self.update(packet);
            } else {
                // Bytes 0..aligned land at their natural packet offset.
                let mask = (1u16 << aligned).wrapping_sub(1);
                let packet_l = _mm_maskz_loadu_epi8(mask as __mmask16, bytes.as_ptr() as *const i8);
                let last3 = super::load3_allow_unordered(remainder, size_mod4);
                let packet_h = _mm_cvtsi64_si128(last3 as i64);
                let packet = _mm256_inserti128_si256(_mm256_castsi128_si256(packet_l), packet_h, 1);
                self.update(packet);
            }
        }
    }

    /// # Safety
    /// CPU must support avx512f + avx512vl + avx512bw.
    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    pub unsafe fn finalize64(&mut self) -> u64 {
        // SAFETY: the caller guarantees the AVX-512 features.
        unsafe {
            for _ in 0..4 {
                let p = permute(self.v0);
                self.update(p);
            }
            let sum0 = _mm256_castsi256_si128(_mm256_add_epi64(self.v0, self.mul0));
            let sum1 = _mm256_castsi256_si128(_mm256_add_epi64(self.v1, self.mul1));
            let hash = _mm_add_epi64(sum0, sum1);
            _mm_cvtsi128_si64(hash) as u64
        }
    }

    /// # Safety
    /// CPU must support avx512f + avx512vl + avx512bw.
    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    pub unsafe fn finalize128(&mut self) -> [u64; 2] {
        // SAFETY: the caller guarantees the AVX-512 features.
        unsafe {
            for _ in 0..6 {
                let p = permute(self.v0);
                self.update(p);
            }
            let sum0 = _mm256_castsi256_si128(_mm256_add_epi64(self.v0, self.mul0));
            let sum1 = _mm256_extracti128_si256(_mm256_add_epi64(self.v1, self.mul1), 1);
            let hash = _mm_add_epi64(sum0, sum1);
            let mut out = [0u64; 2];
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, hash);
            out
        }
    }

    /// # Safety
    /// CPU must support avx512f + avx512vl + avx512bw.
    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    pub unsafe fn finalize256(&mut self) -> [u64; 4] {
        // SAFETY: the caller guarantees the AVX-512 features.
        unsafe {
            for _ in 0..10 {
                let p = permute(self.v0);
                self.update(p);
            }
            let sum0 = _mm256_add_epi64(self.v0, self.mul0);
            let sum1 = _mm256_add_epi64(self.v1, self.mul1);
            let hash = modular_reduction(sum1, sum0);
            let mut out = [0u64; 4];
            _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, hash);
            out
        }
    }
}
