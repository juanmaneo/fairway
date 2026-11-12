//! NEON HighwayHash state for aarch64 (port of `HHStateNEON` from `hh_neon.h`).
//!
//! NEON is always available on aarch64, so methods are safe. Lane order:
//! `V2x64U(a, b)` = `(high=a, low=b)`, i.e. lane 0 is the low 64 bits.

use core::arch::aarch64::*;

use crate::Key;

const INIT0_L: (u64, u64) = (0xa4093822299f31d0, 0xdbe6d5d5fe4cce2f);
const INIT0_H: (u64, u64) = (0x243f6a8885a308d3, 0x13198a2e03707344);
const INIT1_L: (u64, u64) = (0xc0acf169b5f18a8c, 0x3bd39e10cb0ef593);
const INIT1_H: (u64, u64) = (0x452821e638d01377, 0xbe5466cf34e90c6c);

#[inline(always)]
fn v2(a: (u64, u64)) -> uint64x2_t {
    // lane0 = low = a.1, lane1 = high = a.0
    unsafe { vsetq_lane_u64(a.1, vsetq_lane_u64(a.0, vdupq_n_u64(0), 1), 0) }
}

/// Swap 32-bit halves of each 64-bit lane.
#[inline(always)]
fn rotate64_by32(v: uint64x2_t) -> uint64x2_t {
    unsafe { vreinterpretq_u64_u32(vrev64q_u32(vreinterpretq_u32_u64(v))) }
}

#[inline(always)]
fn zipper_merge(v: uint64x2_t) -> uint64x2_t {
    #[rustfmt::skip]
    const POS: [u8; 16] = [3, 12, 2, 5, 14, 1, 15, 0, 11, 4, 10, 13, 9, 6, 8, 7];
    unsafe {
        let tbl = vld1q_u8(POS.as_ptr());
        vreinterpretq_u64_u8(vqtbl1q_u8(vreinterpretq_u8_u64(v), tbl))
    }
}

/// Zero-initialized vector with the lower 0/4/8/12 bytes of `bytes` loaded.
#[inline(always)]
unsafe fn load_multiple_of_four(bytes: *const u8, size: usize) -> uint64x2_t {
    unsafe {
        let words = bytes as *const u32;
        let mask_pattern = [0xFFFFFFFFu64, 0];
        let mut mask4 = vld1q_u64(mask_pattern.as_ptr());
        let mut ret = vdupq_n_u64(0);
        let mut words = words;
        if size & 8 != 0 {
            // Only 8 bytes (upstream `vld1q_low_u64`): the rest may be past
            // the end of the input.
            ret = vcombine_u64(vld1_u64(bytes as *const u64), vdup_n_u64(0));
            // 128-bit shift left by 8 bytes (upstream `vshlq_n_u128(mask4, 8)`):
            // the mask moves to the upper lane.
            mask4 = vextq_u64(vdupq_n_u64(0), mask4, 1);
            words = words.add(2);
        }
        if size & 4 != 0 {
            let w = core::ptr::read_unaligned(words);
            let broadcast = vreinterpretq_u64_u32(vdupq_n_u32(w));
            ret = vorrq_u64(ret, vandq_u64(broadcast, mask4));
        }
        ret
    }
}

/// XORs x << 1 and x << 2 into `out` after clearing the upper two bits of x.
#[inline(always)]
fn xor_by_shift128_left12(x: uint64x2_t, out: &mut uint64x2_t) {
    unsafe {
        let zero64 = vdupq_n_u64(0);
        // 0x80000000 in the top 32-bit lane => lane1 = 0x8000000000000000
        let sign_bit128 = vextq_u64(zero64, vdupq_n_u64(0x8000000000000000), 1);
        let top_bits2 = vshrq_n_u64(x, 62);
        let shifted1_unmasked = vaddq_u64(x, x);
        let top_bits1 = vshrq_n_u64(x, 63);
        let shifted2 = vaddq_u64(shifted1_unmasked, shifted1_unmasked);
        let new_low_bits2 = vextq_u64(zero64, top_bits2, 1);
        *out = veorq_u64(*out, shifted2);
        // andnot: ~sign_bit128 & shifted1_unmasked
        let shifted1 = vbicq_u64(shifted1_unmasked, sign_bit128);
        let new_low_bits1 = vextq_u64(zero64, top_bits1, 1);
        *out = veorq_u64(*out, new_low_bits2);
        *out = veorq_u64(*out, shifted1);
        *out = veorq_u64(*out, new_low_bits1);
    }
}

#[inline(always)]
fn modular_reduction(a32_unmasked: uint64x2_t, a10: uint64x2_t) -> uint64x2_t {
    let mut out = a10;
    xor_by_shift128_left12(a32_unmasked, &mut out);
    out
}

/// NEON hash state (eight 128-bit registers).
#[derive(Clone, Copy)]
pub struct State {
    v0_l: uint64x2_t,
    v0_h: uint64x2_t,
    v1_l: uint64x2_t,
    v1_h: uint64x2_t,
    mul0_l: uint64x2_t,
    mul0_h: uint64x2_t,
    mul1_l: uint64x2_t,
    mul1_h: uint64x2_t,
}

impl State {
    pub fn new(key: Key) -> Self {
        unsafe {
            let key_l = vld1q_u64(key.0.as_ptr());
            let key_h = vld1q_u64(key.0.as_ptr().add(2));
            let (init0_l, init0_h) = (v2(INIT0_L), v2(INIT0_H));
            let (init1_l, init1_h) = (v2(INIT1_L), v2(INIT1_H));
            State {
                v0_l: veorq_u64(key_l, init0_l),
                v0_h: veorq_u64(key_h, init0_h),
                v1_l: veorq_u64(rotate64_by32(key_l), init1_l),
                v1_h: veorq_u64(rotate64_by32(key_h), init1_h),
                mul0_l: init0_l,
                mul0_h: init0_h,
                mul1_l: init1_l,
                mul1_h: init1_h,
            }
        }
    }

    #[inline]
    pub fn update_packet(&mut self, packet: &[u8; 32]) {
        unsafe {
            let packet_l = vld1q_u64(packet.as_ptr() as *const u64);
            let packet_h = vld1q_u64(packet.as_ptr().add(16) as *const u64);
            self.update(packet_h, packet_l);
        }
    }

    #[inline(always)]
    fn update(&mut self, packet_h: uint64x2_t, packet_l: uint64x2_t) {
        unsafe {
            self.v1_l = vaddq_u64(self.v1_l, packet_l);
            self.v1_h = vaddq_u64(self.v1_h, packet_h);
            self.v1_l = vaddq_u64(self.v1_l, self.mul0_l);
            self.v1_h = vaddq_u64(self.v1_h, self.mul0_h);
            self.mul0_l = veorq_u64(
                self.mul0_l,
                vmull_u32(vmovn_u64(self.v1_l), vshrn_n_u64(self.v0_l, 32)),
            );
            self.mul0_h = veorq_u64(
                self.mul0_h,
                vmull_u32(vmovn_u64(self.v1_h), vshrn_n_u64(self.v0_h, 32)),
            );
            self.v0_l = vaddq_u64(self.v0_l, self.mul1_l);
            self.v0_h = vaddq_u64(self.v0_h, self.mul1_h);
            self.mul1_l = veorq_u64(
                self.mul1_l,
                vmull_u32(vmovn_u64(self.v0_l), vshrn_n_u64(self.v1_l, 32)),
            );
            self.mul1_h = veorq_u64(
                self.mul1_h,
                vmull_u32(vmovn_u64(self.v0_h), vshrn_n_u64(self.v1_h, 32)),
            );
            self.v0_l = vaddq_u64(self.v0_l, zipper_merge(self.v1_l));
            self.v0_h = vaddq_u64(self.v0_h, zipper_merge(self.v1_h));
            self.v1_l = vaddq_u64(self.v1_l, zipper_merge(self.v0_l));
            self.v1_h = vaddq_u64(self.v1_h, zipper_merge(self.v0_h));
        }
    }

    pub fn update_remainder(&mut self, bytes: &[u8]) {
        let size_mod32 = bytes.len();
        debug_assert!((1..32).contains(&size_mod32));
        unsafe {
            let vsize = vdupq_n_s32(size_mod32 as i32);
            let shift_right = vdupq_n_s32(size_mod32 as i32 - 32);
            self.v0_l = vaddq_u64(self.v0_l, vreinterpretq_u64_s32(vsize));
            self.v0_h = vaddq_u64(self.v0_h, vreinterpretq_u64_s32(vsize));
            let rot = |v: uint64x2_t| -> uint64x2_t {
                vreinterpretq_u64_u32(vorrq_u32(
                    vshlq_u32(vreinterpretq_u32_u64(v), vsize),
                    vshlq_u32(vreinterpretq_u32_u64(v), shift_right),
                ))
            };
            self.v1_l = rot(self.v1_l);
            self.v1_h = rot(self.v1_h);

            let size_mod4 = size_mod32 & 3;
            let remainder = bytes.as_ptr().add(size_mod32 & !3);

            if size_mod32 & 16 != 0 {
                let packet_l = vld1q_u64(bytes.as_ptr() as *const u64);
                let mut packet_h = load_multiple_of_four(bytes.as_ptr().add(16), size_mod32);
                let last4 =
                    crate::arch::shared::load3_allow_read_before_and_return(remainder, size_mod4);
                packet_h = vreinterpretq_u64_u32(vsetq_lane_u32(
                    last4,
                    vreinterpretq_u32_u64(packet_h),
                    3,
                ));
                self.update(packet_h, packet_l);
            } else {
                let packet_l = load_multiple_of_four(bytes.as_ptr(), size_mod32);
                let last4 = crate::arch::shared::load3_allow_unordered(remainder, size_mod4);
                let packet_h = vld1q_u64([last4, 0].as_ptr());
                self.update(packet_h, packet_l);
            }
        }
    }

    #[inline(always)]
    fn permute_and_update(&mut self) {
        self.update(rotate64_by32(self.v0_l), rotate64_by32(self.v0_h));
    }

    pub fn finalize64(&mut self) -> u64 {
        for _ in 0..4 {
            self.permute_and_update();
        }
        unsafe {
            let sum0 = vaddq_u64(self.v0_l, self.mul0_l);
            let sum1 = vaddq_u64(self.v1_l, self.mul1_l);
            let hash = vaddq_u64(sum0, sum1);
            vgetq_lane_u64(hash, 0)
        }
    }

    pub fn finalize128(&mut self) -> [u64; 2] {
        for _ in 0..6 {
            self.permute_and_update();
        }
        unsafe {
            let sum0 = vaddq_u64(self.v0_l, self.mul0_l);
            let sum1 = vaddq_u64(self.v1_h, self.mul1_h);
            let hash = vaddq_u64(sum0, sum1);
            [vgetq_lane_u64(hash, 0), vgetq_lane_u64(hash, 1)]
        }
    }

    pub fn finalize256(&mut self) -> [u64; 4] {
        for _ in 0..10 {
            self.permute_and_update();
        }
        unsafe {
            let sum0_l = vaddq_u64(self.v0_l, self.mul0_l);
            let sum1_l = vaddq_u64(self.v1_l, self.mul1_l);
            let sum0_h = vaddq_u64(self.v0_h, self.mul0_h);
            let sum1_h = vaddq_u64(self.v1_h, self.mul1_h);
            let hash_l = modular_reduction(sum1_l, sum0_l);
            let hash_h = modular_reduction(sum1_h, sum0_h);
            [
                vgetq_lane_u64(hash_l, 0),
                vgetq_lane_u64(hash_l, 1),
                vgetq_lane_u64(hash_h, 0),
                vgetq_lane_u64(hash_h, 1),
            ]
        }
    }
}
