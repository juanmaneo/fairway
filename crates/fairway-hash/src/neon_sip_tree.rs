//! NEON SipTreeHash for aarch64.
//!
//! There is no upstream NEON tree hash (`sip_tree_hash.cc` has an AVX2 and a
//! scalar version only), so this is not a port: it runs the four SipHash
//! chains of the tree hash on 128-bit registers, two chains per register
//! (each state word is a pair of `uint64x2_t`: chains 0-1 and chains 2-3),
//! and produces the same value as the portable `tree_compute_scalar` in
//! `sip.rs`.
//!
//! NEON is always available on aarch64, so everything here is safe to call.
//! Packets are loaded as little-endian bytes, which is what the reference
//! defines; the module is therefore only built for little-endian targets.

use core::arch::aarch64::*;

use crate::sip::{final_packet, reduce, TREE_LANES};
use crate::{Key, SipHashKey};

/// One SipHash state word of the four chains: chains 0-1 and chains 2-3.
#[derive(Clone, Copy)]
struct Word([uint64x2_t; 2]);

impl Word {
    #[inline(always)]
    fn splat(value: u64) -> Self {
        // SAFETY: NEON is always available on aarch64.
        let v = unsafe { vdupq_n_u64(value) };
        Word([v, v])
    }

    /// The four 64-bit chains from `words`.
    #[inline(always)]
    fn load(words: &[u64; TREE_LANES]) -> Self {
        // SAFETY: NEON is always available on aarch64; the loads read exactly
        // the 32 bytes of `words`.
        unsafe { Word([vld1q_u64(words.as_ptr()), vld1q_u64(words.as_ptr().add(2))]) }
    }

    /// The four 64-bit chains from a 32-byte packet (little-endian words).
    #[inline(always)]
    fn load_packet(packet: &[u8; 32]) -> Self {
        // SAFETY: NEON is always available on aarch64; the loads read exactly
        // the 32 bytes of `packet`, and the byte-wise load has no alignment
        // requirement.
        unsafe {
            Word([
                vreinterpretq_u64_u8(vld1q_u8(packet.as_ptr())),
                vreinterpretq_u64_u8(vld1q_u8(packet.as_ptr().add(16))),
            ])
        }
    }

    #[inline(always)]
    fn store(self) -> [u64; TREE_LANES] {
        let mut words = [0u64; TREE_LANES];
        // SAFETY: NEON is always available on aarch64; the stores write
        // exactly the 32 bytes of `words`.
        unsafe {
            vst1q_u64(words.as_mut_ptr(), self.0[0]);
            vst1q_u64(words.as_mut_ptr().add(2), self.0[1]);
        }
        words
    }

    #[inline(always)]
    fn xor(self, other: Word) -> Word {
        // SAFETY: NEON is always available on aarch64.
        unsafe {
            Word([
                veorq_u64(self.0[0], other.0[0]),
                veorq_u64(self.0[1], other.0[1]),
            ])
        }
    }

    #[inline(always)]
    fn add(self, other: Word) -> Word {
        // SAFETY: NEON is always available on aarch64.
        unsafe {
            Word([
                vaddq_u64(self.0[0], other.0[0]),
                vaddq_u64(self.0[1], other.0[1]),
            ])
        }
    }

    /// Rotates each 64-bit chain left by 13 bits: a shift, then a
    /// shift-right-and-insert that merges the bits shifted out.
    #[inline(always)]
    fn rotate_left13(self) -> Word {
        // SAFETY: NEON is always available on aarch64.
        unsafe { Word(self.0.map(|v| vsriq_n_u64::<51>(vshlq_n_u64::<13>(v), v))) }
    }

    /// Rotates each 64-bit chain left by 16 bits.
    #[inline(always)]
    fn rotate_left16(self) -> Word {
        // SAFETY: NEON is always available on aarch64.
        unsafe { Word(self.0.map(|v| vsriq_n_u64::<48>(vshlq_n_u64::<16>(v), v))) }
    }

    /// Rotates each 64-bit chain left by 17 bits.
    #[inline(always)]
    fn rotate_left17(self) -> Word {
        // SAFETY: NEON is always available on aarch64.
        unsafe { Word(self.0.map(|v| vsriq_n_u64::<47>(vshlq_n_u64::<17>(v), v))) }
    }

    /// Rotates each 64-bit chain left by 21 bits.
    #[inline(always)]
    fn rotate_left21(self) -> Word {
        // SAFETY: NEON is always available on aarch64.
        unsafe { Word(self.0.map(|v| vsriq_n_u64::<43>(vshlq_n_u64::<21>(v), v))) }
    }

    /// Rotates each 64-bit chain left by 32 bits: swaps its two 32-bit
    /// halves.
    #[inline(always)]
    fn rotate_left32(self) -> Word {
        // SAFETY: NEON is always available on aarch64.
        unsafe {
            Word(
                self.0
                    .map(|v| vreinterpretq_u64_u32(vrev64q_u32(vreinterpretq_u32_u64(v)))),
            )
        }
    }
}

/// The four SipHash states, one per chain.
struct State<const UPDATE_ROUNDS: usize, const FINALIZE_ROUNDS: usize> {
    v0: Word,
    v1: Word,
    v2: Word,
    v3: Word,
}

impl<const U: usize, const F: usize> State<U, F> {
    /// Chain `i` is the SipHash state keyed with `key[i] ^ (4 | i)`, as in
    /// the scalar tree hash.
    #[inline(always)]
    fn new(key: Key) -> Self {
        let lane_keys: [u64; TREE_LANES] =
            core::array::from_fn(|lane| key.0[lane] ^ (TREE_LANES | lane) as u64);
        let lane_keys = Word::load(&lane_keys);
        State {
            v0: Word::splat(0x736f6d6570736575).xor(lane_keys),
            v1: Word::splat(0x646f72616e646f6d).xor(lane_keys),
            v2: Word::splat(0x6c7967656e657261).xor(lane_keys),
            v3: Word::splat(0x7465646279746573).xor(lane_keys),
        }
    }

    #[inline(always)]
    fn update(&mut self, packet: Word) {
        self.v3 = self.v3.xor(packet);
        self.compress::<U>();
        self.v0 = self.v0.xor(packet);
    }

    /// The four chain hashes.
    #[inline(always)]
    fn finalize(mut self) -> Word {
        // Mix in bits to avoid leaking the key if all packets were zero.
        self.v2 = self.v2.xor(Word::splat(0xFF));
        self.compress::<F>();
        self.v0.xor(self.v1).xor(self.v2.xor(self.v3))
    }

    #[inline(always)]
    fn compress<const ROUNDS: usize>(&mut self) {
        for _ in 0..ROUNDS {
            // ARX network: add, rotate, exclusive-or.
            self.v0 = self.v0.add(self.v1);
            self.v2 = self.v2.add(self.v3);
            self.v1 = self.v1.rotate_left13().xor(self.v0);
            self.v3 = self.v3.rotate_left16().xor(self.v2);

            self.v0 = self.v0.rotate_left32();

            self.v2 = self.v2.add(self.v1);
            self.v0 = self.v0.add(self.v3);
            self.v1 = self.v1.rotate_left17().xor(self.v2);
            self.v3 = self.v3.rotate_left21().xor(self.v0);

            self.v2 = self.v2.rotate_left32();
        }
    }
}

/// SipTreeHash with `U` update and `F` finalization rounds, equal to the
/// portable `tree_compute_scalar::<U, F>`.
pub(crate) fn tree_hash<const U: usize, const F: usize>(key: Key, bytes: &[u8]) -> u64 {
    let (packets, tail) = bytes.as_chunks::<32>();
    let mut state = State::<U, F>::new(key);
    for packet in packets {
        state.update(Word::load_packet(packet));
    }
    // NEON has no masked load, so the final packet is built in a buffer.
    state.update(Word::load_packet(&final_packet(tail)));
    let hashes = state.finalize().store();
    // The four chain hashes are reduced to one with a scalar SipHash.
    reduce::<U, F>(SipHashKey([key.0[0], key.0[1]]), &hashes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sip::{scalar_sip_tree_hash, scalar_sip_tree_hash13};
    use crate::vectors::{SIP_TREE_HASH13_OUTPUT, SIP_TREE_HASH_OUTPUT, SIP_TREE_KEY};

    #[test]
    fn neon_tree_hash_golden() {
        let input: Vec<u8> = (0..65536).map(|i| i as u8).collect();
        let key = Key(SIP_TREE_KEY);
        for (size, expected) in SIP_TREE_HASH_OUTPUT {
            assert_eq!(
                tree_hash::<2, 4>(key, &input[..size]),
                expected,
                "neon tree24 at {size}"
            );
        }
        for (size, expected) in SIP_TREE_HASH13_OUTPUT {
            assert_eq!(
                tree_hash::<1, 3>(key, &input[..size]),
                expected,
                "neon tree13 at {size}"
            );
        }
    }

    /// Every length 0..=160 (the final packet's 32 tail lengths, with up to
    /// five whole packets before it) against the portable code.
    #[test]
    fn neon_tree_hash_matches_scalar() {
        let input: Vec<u8> = (0..160u32).map(|i| (i * 7 + 3) as u8).collect();
        let key = Key(SIP_TREE_KEY);
        for len in 0..=input.len() {
            let bytes = &input[..len];
            assert_eq!(
                tree_hash::<2, 4>(key, bytes),
                scalar_sip_tree_hash(key, bytes),
                "neon tree24 at {len}"
            );
            assert_eq!(
                tree_hash::<1, 3>(key, bytes),
                scalar_sip_tree_hash13(key, bytes),
                "neon tree13 at {len}"
            );
        }
    }
}
