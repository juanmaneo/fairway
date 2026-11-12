//! Portable scalar HighwayHash (direct port of `hh_portable.h`).
//! Works on any platform, including big-endian (loads are little-endian).

use crate::Key;

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
const fn rotate64_by32(x: u64) -> u64 {
    x.rotate_right(32)
}

#[inline(always)]
const fn mask(v: u64, bytes: u64) -> u64 {
    v & (0xFFu64 << (bytes * 8))
}

/// Zipper-merge of two 64-bit lanes; returns the amounts to add to
/// `(add0, add1)`.
#[inline(always)]
fn zipper_merge(v1: u64, v0: u64) -> (u64, u64) {
    let add0 = ((mask(v0, 3) + mask(v1, 4)) >> 24)
        + ((mask(v0, 5) + mask(v1, 6)) >> 16)
        + mask(v0, 2)
        + (mask(v0, 1) << 32)
        + (mask(v1, 7) >> 8)
        + (v0 << 56);
    let add1 = ((mask(v1, 3) + mask(v0, 4)) >> 24)
        + mask(v1, 2)
        + (mask(v1, 5) >> 16)
        + (mask(v1, 1) << 24)
        + (mask(v0, 6) >> 8)
        + (mask(v1, 0) << 48)
        + mask(v0, 7);
    (add0, add1)
}

/// Loads 0..=3 bytes; the bytes need not be in little-endian order.
/// This order is part of the HighwayHash length-padding definition.
#[inline(always)]
fn load3_allow_unordered(from: &[u8], size_mod4: usize) -> u64 {
    if size_mod4 == 0 {
        return 0;
    }
    let idx0 = 0;
    let idx1 = size_mod4 >> 1;
    let idx2 = size_mod4 - 1;
    (from[idx0] as u64) + ((from[idx1] as u64) << 8) + ((from[idx2] as u64) << 16)
}

/// Up to 4 preceding bytes may be read and returned; valid bytes are in
/// little-endian order in the low bytes. `bytes` is the full remainder
/// (length `size_mod32`); the 4 loaded bytes end at
/// `(size_mod32 & !3) + size_mod4`, which is always within `bytes`.
#[inline(always)]
fn load3_allow_read_before_and_return(bytes: &[u8], size_mod32: usize, size_mod4: usize) -> u32 {
    let end = (size_mod32 & !3) + size_mod4;
    let start = end - 4;
    u32::from_le_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ])
}

/// Hash state; mirrors `HHStatePortable`.
#[derive(Clone)]
pub struct State {
    v0: [u64; 4],
    v1: [u64; 4],
    mul0: [u64; 4],
    mul1: [u64; 4],
}

impl State {
    pub fn new(key: Key) -> Self {
        let mut s = State {
            v0: [0; 4],
            v1: [0; 4],
            mul0: INIT0,
            mul1: INIT1,
        };
        s.reset(key);
        s
    }

    pub fn reset(&mut self, key: Key) {
        let rotated = [
            rotate64_by32(key.0[0]),
            rotate64_by32(key.0[1]),
            rotate64_by32(key.0[2]),
            rotate64_by32(key.0[3]),
        ];
        for lane in 0..4 {
            self.v0[lane] = INIT0[lane] ^ key.0[lane];
            self.v1[lane] = INIT1[lane] ^ rotated[lane];
        }
        self.mul0 = INIT0;
        self.mul1 = INIT1;
    }

    /// One 32-byte packet.
    #[inline]
    pub fn update_packet(&mut self, packet: &[u8; 32]) {
        let (words, _) = packet.as_chunks::<8>();
        self.update_lanes(core::array::from_fn(|i| u64::from_le_bytes(words[i])));
    }

    #[inline(always)]
    fn update_lanes(&mut self, packet: [u64; 4]) {
        for (lane, p) in packet.into_iter().enumerate() {
            self.v1[lane] = self.v1[lane].wrapping_add(p).wrapping_add(self.mul0[lane]);
        }
        for lane in 0..4 {
            let v1_32 = self.v1[lane] as u32;
            self.mul0[lane] ^= (v1_32 as u64).wrapping_mul(self.v0[lane] >> 32);
            self.v0[lane] = self.v0[lane].wrapping_add(self.mul1[lane]);
            let v0_32 = self.v0[lane] as u32;
            self.mul1[lane] ^= (v0_32 as u64).wrapping_mul(self.v1[lane] >> 32);
        }
        let (v0, v1) = (&mut self.v0, &mut self.v1);
        let (a0, a1) = zipper_merge(v1[1], v1[0]);
        v0[0] = v0[0].wrapping_add(a0);
        v0[1] = v0[1].wrapping_add(a1);
        let (a0, a1) = zipper_merge(v1[3], v1[2]);
        v0[2] = v0[2].wrapping_add(a0);
        v0[3] = v0[3].wrapping_add(a1);
        let (a0, a1) = zipper_merge(v0[1], v0[0]);
        v1[0] = v1[0].wrapping_add(a0);
        v1[1] = v1[1].wrapping_add(a1);
        let (a0, a1) = zipper_merge(v0[3], v0[2]);
        v1[2] = v1[2].wrapping_add(a0);
        v1[3] = v1[3].wrapping_add(a1);
    }

    /// Final partial packet of `size_mod32` bytes (1..=31).
    #[inline]
    pub fn update_remainder(&mut self, bytes: &[u8]) {
        let size_mod32 = bytes.len();
        debug_assert!((1..32).contains(&size_mod32));
        let mod32_pair = ((size_mod32 as u64) << 32) + size_mod32 as u64;
        for lane in &mut self.v0 {
            *lane = lane.wrapping_add(mod32_pair);
        }
        // Rotate each 32-bit half of v1 left by size_mod32 bits.
        for lane in &mut self.v1 {
            let lo = *lane as u32;
            let hi = (*lane >> 32) as u32;
            let c = size_mod32 as u32;
            let lo_r = lo.rotate_left(c);
            let hi_r = hi.rotate_left(c);
            *lane = ((hi_r as u64) << 32) | lo_r as u64;
        }

        let size_mod4 = size_mod32 & 3;
        let remainder = &bytes[size_mod32 & !3..];

        let mut packet = [0u8; 32];
        packet[..size_mod32 & !3].copy_from_slice(&bytes[..size_mod32 & !3]);

        if size_mod32 & 16 != 0 {
            // 16..31 bytes left: read the last 0..3 bytes plus up to 4
            // preceding bytes into the upper four bytes of the packet.
            let last4 = load3_allow_read_before_and_return(bytes, size_mod32, size_mod4);
            packet[28..32].copy_from_slice(&last4.to_le_bytes());
        } else {
            // < 16 bytes left: initialize the otherwise empty upper half
            // with up to 64 bits of padding.
            let last4 = load3_allow_unordered(remainder, size_mod4);
            packet[16..24].copy_from_slice(&last4.to_le_bytes());
        }
        self.update_packet(&packet);
    }

    #[inline]
    fn permute_and_update(&mut self) {
        let permuted = [
            rotate64_by32(self.v0[2]),
            rotate64_by32(self.v0[3]),
            rotate64_by32(self.v0[0]),
            rotate64_by32(self.v0[1]),
        ];
        self.update_lanes(permuted);
    }

    pub fn finalize64(&mut self) -> u64 {
        for _ in 0..4 {
            self.permute_and_update();
        }
        self.v0[0]
            .wrapping_add(self.v1[0])
            .wrapping_add(self.mul0[0])
            .wrapping_add(self.mul1[0])
    }

    pub fn finalize128(&mut self) -> [u64; 2] {
        for _ in 0..6 {
            self.permute_and_update();
        }
        [
            self.v0[0]
                .wrapping_add(self.mul0[0])
                .wrapping_add(self.v1[2])
                .wrapping_add(self.mul1[2]),
            self.v0[1]
                .wrapping_add(self.mul0[1])
                .wrapping_add(self.v1[3])
                .wrapping_add(self.mul1[3]),
        ]
    }

    pub fn finalize256(&mut self) -> [u64; 4] {
        for _ in 0..10 {
            self.permute_and_update();
        }
        let (v0, v1, mul0, mul1) = (&self.v0, &self.v1, &self.mul0, &self.mul1);
        let (m1, m0) = modular_reduction(
            v1[1].wrapping_add(mul1[1]),
            v1[0].wrapping_add(mul1[0]),
            v0[1].wrapping_add(mul0[1]),
            v0[0].wrapping_add(mul0[0]),
        );
        let (m3, m2) = modular_reduction(
            v1[3].wrapping_add(mul1[3]),
            v1[2].wrapping_add(mul1[2]),
            v0[3].wrapping_add(mul0[3]),
            v0[2].wrapping_add(mul0[2]),
        );
        [m0, m1, m2, m3]
    }
}

/// a << kBits for 128-bit a = (a1, a0).
#[inline(always)]
fn shift128_left<const KBITS: u32>(a1: &mut u64, a0: &mut u64) {
    let shifted1 = *a1 << KBITS;
    let top_bits = *a0 >> (64 - KBITS);
    *a0 <<= KBITS;
    *a1 = shifted1 | top_bits;
}

/// Modular reduction by the irreducible polynomial x^128 + x^2 + x.
/// Input: a 256-bit number a3210.
#[inline(always)]
fn modular_reduction(a3_unmasked: u64, a2: u64, a1: u64, a0: u64) -> (u64, u64) {
    let a3 = a3_unmasked & 0x3FFF_FFFF_FFFF_FFFF;
    let (mut a3_shl1, mut a2_shl1) = (a3, a2);
    let (mut a3_shl2, mut a2_shl2) = (a3, a2);
    shift128_left::<1>(&mut a3_shl1, &mut a2_shl1);
    shift128_left::<2>(&mut a3_shl2, &mut a2_shl2);
    (a1 ^ a3_shl1 ^ a3_shl2, a0 ^ a2_shl1 ^ a2_shl2)
}

/// One-shot 64-bit hash with the portable backend.
pub fn hash64(key: Key, bytes: &[u8]) -> u64 {
    let mut state = State::new(key);
    hash_into(&mut state, bytes);
    state.finalize64()
}

/// One-shot 128-bit hash with the portable backend.
pub fn hash128(key: Key, bytes: &[u8]) -> [u64; 2] {
    let mut state = State::new(key);
    hash_into(&mut state, bytes);
    state.finalize128()
}

/// One-shot 256-bit hash with the portable backend.
pub fn hash256(key: Key, bytes: &[u8]) -> [u64; 4] {
    let mut state = State::new(key);
    hash_into(&mut state, bytes);
    state.finalize256()
}

fn hash_into(state: &mut State, bytes: &[u8]) {
    let (chunks, rest) = bytes.as_chunks::<32>();
    for chunk in chunks {
        state.update_packet(chunk);
    }
    if !rest.is_empty() {
        state.update_remainder(rest);
    }
}
