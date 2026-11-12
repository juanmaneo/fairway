//! SSE4.1 HighwayHash state (port of `HHStateSSE41` from `hh_sse41.h`).
//! Uses pairs of 128-bit lanes to emulate the AVX2 algorithm.
//!
//! All methods are `unsafe` because they execute SSE4.1 instructions;
//! callers must ensure the CPU supports SSE4.1 (e.g.
//! `is_x86_feature_detected!("sse4.1")`).

use core::arch::x86_64::*;

use crate::Key;

/// "Nothing up my sleeve" numbers; lane order matches the C++ V2x64U
/// constructors: (high, low) per 128-bit lane.
const INIT0_L: (u64, u64) = (0xa4093822299f31d0, 0xdbe6d5d5fe4cce2f);
const INIT0_H: (u64, u64) = (0x243f6a8885a308d3, 0x13198a2e03707344);
const INIT1_L: (u64, u64) = (0xc0acf169b5f18a8c, 0x3bd39e10cb0ef593);
const INIT1_H: (u64, u64) = (0x452821e638d01377, 0xbe5466cf34e90c6c);

#[inline(always)]
unsafe fn v2(a: (u64, u64)) -> __m128i {
    unsafe { _mm_set_epi64x(a.0 as i64, a.1 as i64) }
}

/// Swap 32-bit halves of each 64-bit lane.
#[inline(always)]
unsafe fn rotate64_by32(v: __m128i) -> __m128i {
    unsafe { _mm_shuffle_epi32(v, 0b10_11_00_01) }
}

/// Rotates 32-bit lanes by `count` bits.
#[inline(always)]
unsafe fn rotate32_by(v: __m128i, count: i64) -> __m128i {
    unsafe {
        let cl = _mm_cvtsi64_si128(count);
        let cr = _mm_cvtsi64_si128(32 - count);
        let left = _mm_sll_epi32(v, cl);
        let right = _mm_srl_epi32(v, cr);
        _mm_or_si128(left, right)
    }
}

#[inline(always)]
unsafe fn zipper_merge(v: __m128i) -> __m128i {
    unsafe {
        // Desired byte permutation (output byte j <- input byte perm[j]):
        //   lane0: 3 12 2 5 14 1 15 0, lane1: 11 4 10 13 9 6 8 7
        let lane0 = 0x000F010E05020C03u64;
        let lane1 = 0x070806090D0A040Bu64;
        let ctrl = _mm_set_epi64x(lane1 as i64, lane0 as i64);
        _mm_shuffle_epi8(v, ctrl)
    }
}

/// Zero-initialized vector with the lower 0/4/8/12 bytes of `bytes` loaded
/// (replacement for AVX2 maskload).
#[inline(always)]
unsafe fn load_multiple_of_four(bytes: *const u8, size: usize) -> __m128i {
    unsafe {
        let words = bytes as *const u32;
        let mut mask4 = _mm_cvtsi64_si128(0xFFFFFFFFu64 as i64);
        let mut ret = _mm_setzero_si128();
        let mut words = words;
        if size & 8 != 0 {
            ret = _mm_loadl_epi64(words as *const __m128i);
            mask4 = _mm_slli_si128(mask4, 8);
            words = words.add(2);
        }
        if size & 4 != 0 {
            let w = core::ptr::read_unaligned(words);
            let broadcast = _mm_shuffle_epi32(_mm_cvtsi32_si128(w as i32), 0x00);
            ret = _mm_or_si128(ret, _mm_and_si128(broadcast, mask4));
        }
        ret
    }
}

/// XORs x << 1 and x << 2 into `out` after clearing the upper two bits of x.
#[inline(always)]
unsafe fn xor_by_shift128_left12(x: __m128i, out: &mut __m128i) {
    unsafe {
        let zero = _mm_setzero_si128();
        let sign_bit128 = _mm_insert_epi32(zero, 0x80000000u32 as i32, 3);
        let top_bits2 = _mm_srli_epi64(x, 64 - 2);
        let shifted1_unmasked = _mm_add_epi64(x, x);
        let top_bits1 = _mm_srli_epi64(x, 64 - 1);
        let shifted2 = _mm_add_epi64(shifted1_unmasked, shifted1_unmasked);
        let new_low_bits2 = _mm_slli_si128(top_bits2, 8);
        *out = _mm_xor_si128(*out, shifted2);
        let shifted1 = _mm_andnot_si128(sign_bit128, shifted1_unmasked);
        let new_low_bits1 = _mm_slli_si128(top_bits1, 8);
        *out = _mm_xor_si128(*out, new_low_bits2);
        *out = _mm_xor_si128(*out, shifted1);
        *out = _mm_xor_si128(*out, new_low_bits1);
    }
}

/// Modular reduction by x^128 + x^2 + x.
#[inline(always)]
unsafe fn modular_reduction(a32_unmasked: __m128i, a10: __m128i) -> __m128i {
    let mut out = a10;
    unsafe { xor_by_shift128_left12(a32_unmasked, &mut out) };
    out
}

/// SSE4.1 hash state (eight 128-bit registers).
#[derive(Clone, Copy)]
pub struct State {
    v0_l: __m128i,
    v0_h: __m128i,
    v1_l: __m128i,
    v1_h: __m128i,
    mul0_l: __m128i,
    mul0_h: __m128i,
    mul1_l: __m128i,
    mul1_h: __m128i,
}

impl State {
    /// # Safety
    /// CPU must support SSE4.1.
    pub unsafe fn new(key: Key) -> Self {
        unsafe {
            let key_l = _mm_loadu_si128(key.0.as_ptr() as *const __m128i);
            let key_h = _mm_loadu_si128(key.0.as_ptr().add(2) as *const __m128i);
            let (init0_l, init0_h) = (v2(INIT0_L), v2(INIT0_H));
            let (init1_l, init1_h) = (v2(INIT1_L), v2(INIT1_H));
            State {
                v0_l: _mm_xor_si128(key_l, init0_l),
                v0_h: _mm_xor_si128(key_h, init0_h),
                v1_l: _mm_xor_si128(rotate64_by32(key_l), init1_l),
                v1_h: _mm_xor_si128(rotate64_by32(key_h), init1_h),
                mul0_l: init0_l,
                mul0_h: init0_h,
                mul1_l: init1_l,
                mul1_h: init1_h,
            }
        }
    }

    /// # Safety
    /// CPU must support SSE4.1.
    #[inline]
    pub unsafe fn update_packet(&mut self, packet: &[u8; 32]) {
        unsafe {
            let packet_l = _mm_loadu_si128(packet.as_ptr() as *const __m128i);
            let packet_h = _mm_loadu_si128(packet.as_ptr().add(16) as *const __m128i);
            self.update(packet_h, packet_l);
        }
    }

    #[inline(always)]
    unsafe fn update(&mut self, packet_h: __m128i, packet_l: __m128i) {
        unsafe {
            self.v1_l = _mm_add_epi64(self.v1_l, packet_l);
            self.v1_h = _mm_add_epi64(self.v1_h, packet_h);
            self.v1_l = _mm_add_epi64(self.v1_l, self.mul0_l);
            self.v1_h = _mm_add_epi64(self.v1_h, self.mul0_h);
            self.mul0_l = _mm_xor_si128(
                self.mul0_l,
                _mm_mul_epu32(self.v1_l, rotate64_by32(self.v0_l)),
            );
            self.mul0_h = _mm_xor_si128(
                self.mul0_h,
                _mm_mul_epu32(self.v1_h, _mm_srli_epi64(self.v0_h, 32)),
            );
            self.v0_l = _mm_add_epi64(self.v0_l, self.mul1_l);
            self.v0_h = _mm_add_epi64(self.v0_h, self.mul1_h);
            self.mul1_l = _mm_xor_si128(
                self.mul1_l,
                _mm_mul_epu32(self.v0_l, rotate64_by32(self.v1_l)),
            );
            self.mul1_h = _mm_xor_si128(
                self.mul1_h,
                _mm_mul_epu32(self.v0_h, _mm_srli_epi64(self.v1_h, 32)),
            );
            self.v0_l = _mm_add_epi64(self.v0_l, zipper_merge(self.v1_l));
            self.v0_h = _mm_add_epi64(self.v0_h, zipper_merge(self.v1_h));
            self.v1_l = _mm_add_epi64(self.v1_l, zipper_merge(self.v0_l));
            self.v1_h = _mm_add_epi64(self.v1_h, zipper_merge(self.v0_h));
        }
    }

    /// # Safety
    /// CPU must support SSE4.1. `bytes.len()` must be in 1..=31.
    pub unsafe fn update_remainder(&mut self, bytes: &[u8]) {
        let size_mod32 = bytes.len();
        debug_assert!((1..32).contains(&size_mod32));
        unsafe {
            let vsize = _mm_shuffle_epi32(_mm_set1_epi32(size_mod32 as i32), 0b00_00_00_00);
            self.v0_l = _mm_add_epi64(self.v0_l, vsize);
            self.v0_h = _mm_add_epi64(self.v0_h, vsize);
            let c = size_mod32 as i64;
            self.v1_l = rotate32_by(self.v1_l, c);
            self.v1_h = rotate32_by(self.v1_h, c);

            let size_mod4 = size_mod32 & 3;
            let remainder = bytes.as_ptr().add(size_mod32 & !3);

            if size_mod32 & 16 != 0 {
                let packet_l = _mm_loadu_si128(bytes.as_ptr() as *const __m128i);
                let mut packet_h = load_multiple_of_four(bytes.as_ptr().add(16), size_mod32);
                let last4 = super::load3_allow_read_before_and_return(remainder, size_mod4);
                packet_h = _mm_insert_epi32(packet_h, last4 as i32, 3);
                self.update(packet_h, packet_l);
            } else {
                let packet_l = load_multiple_of_four(bytes.as_ptr(), size_mod32);
                let last4 = super::load3_allow_unordered(remainder, size_mod4);
                let packet_h = _mm_cvtsi64_si128(last4 as i64);
                self.update(packet_h, packet_l);
            }
        }
    }

    #[inline(always)]
    unsafe fn permute_and_update(&mut self) {
        unsafe {
            self.update(rotate64_by32(self.v0_l), rotate64_by32(self.v0_h));
        }
    }

    /// # Safety
    /// CPU must support SSE4.1.
    pub unsafe fn finalize64(&mut self) -> u64 {
        unsafe {
            for _ in 0..4 {
                self.permute_and_update();
            }
            let sum0 = _mm_add_epi64(self.v0_l, self.mul0_l);
            let sum1 = _mm_add_epi64(self.v1_l, self.mul1_l);
            let hash = _mm_add_epi64(sum0, sum1);
            _mm_cvtsi128_si64(hash) as u64
        }
    }

    /// # Safety
    /// CPU must support SSE4.1.
    pub unsafe fn finalize128(&mut self) -> [u64; 2] {
        unsafe {
            for _ in 0..6 {
                self.permute_and_update();
            }
            let sum0 = _mm_add_epi64(self.v0_l, self.mul0_l);
            let sum1 = _mm_add_epi64(self.v1_h, self.mul1_h);
            let hash = _mm_add_epi64(sum0, sum1);
            let mut out = [0u64; 2];
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, hash);
            out
        }
    }

    /// # Safety
    /// CPU must support SSE4.1.
    pub unsafe fn finalize256(&mut self) -> [u64; 4] {
        unsafe {
            for _ in 0..10 {
                self.permute_and_update();
            }
            let sum0_l = _mm_add_epi64(self.v0_l, self.mul0_l);
            let sum1_l = _mm_add_epi64(self.v1_l, self.mul1_l);
            let sum0_h = _mm_add_epi64(self.v0_h, self.mul0_h);
            let sum1_h = _mm_add_epi64(self.v1_h, self.mul1_h);
            let hash_l = modular_reduction(sum1_l, sum0_l);
            let hash_h = modular_reduction(sum1_h, sum0_h);
            let mut out = [0u64; 4];
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, hash_l);
            _mm_storeu_si128(out.as_mut_ptr().add(2) as *mut __m128i, hash_h);
            out
        }
    }
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
    fn sse41_golden() {
        if !super::super::has_sse41() {
            return;
        }
        let in_buf = inputs();
        let key = Key(KEY64);
        for size in 0..=64usize {
            let mut s = unsafe { State::new(key) };
            let (chunks, rest) = in_buf[..size].as_chunks::<32>();
            for c in chunks {
                unsafe { s.update_packet(c) };
            }
            if !rest.is_empty() {
                unsafe { s.update_remainder(rest) };
            }
            assert_eq!(
                unsafe { s.finalize64() },
                EXPECTED64[size],
                "sse41 64-bit at {size}"
            );
            let mut s = unsafe { State::new(key) };
            let (chunks, rest) = in_buf[..size].as_chunks::<32>();
            for c in chunks {
                unsafe { s.update_packet(c) };
            }
            if !rest.is_empty() {
                unsafe { s.update_remainder(rest) };
            }
            assert_eq!(
                unsafe { s.finalize128() },
                EXPECTED128[size],
                "sse41 128-bit at {size}"
            );
            let mut s = unsafe { State::new(key) };
            let (chunks, rest) = in_buf[..size].as_chunks::<32>();
            for c in chunks {
                unsafe { s.update_packet(c) };
            }
            if !rest.is_empty() {
                unsafe { s.update_remainder(rest) };
            }
            assert_eq!(
                unsafe { s.finalize256() },
                EXPECTED256[size],
                "sse41 256-bit at {size}"
            );
        }
    }
}
