//! Portable SipHash-2-4 and SipHash-1-3 (port of `sip_hash.h`), plus the
//! j-lanes tree hash extension (port of `scalar_sip_tree_hash.cc`). The tree
//! hash also has SIMD versions: AVX2 (port of `sip_tree_hash.cc`) and, behind
//! the `avx512` feature, AVX-512VL on x86_64, selected at runtime, and NEON on
//! aarch64. The last two have no upstream counterpart.
//!
//! Outputs are identical to the reference SipHash for messages of any length,
//! and every tree hash implementation produces the same value.

use crate::Key;

/// 128-bit SipHash key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SipHashKey(pub [u64; 2]);

impl From<[u64; 2]> for SipHashKey {
    fn from(k: [u64; 2]) -> Self {
        SipHashKey(k)
    }
}

struct SipHashState<const UPDATE_ITERS: usize, const FINALIZE_ITERS: usize> {
    v0: u64,
    v1: u64,
    v2: u64,
    v3: u64,
}

impl<const U: usize, const F: usize> SipHashState<U, F> {
    fn new(key: [u64; 2]) -> Self {
        SipHashState {
            v0: 0x736f6d6570736575 ^ key[0],
            v1: 0x646f72616e646f6d ^ key[1],
            v2: 0x6c7967656e657261 ^ key[0],
            v3: 0x7465646279746573 ^ key[1],
        }
    }

    /// The per-lane state of the tree hash: one key word, tagged with the
    /// number of lanes and the lane index.
    fn new_tree_lane(lane_key: u64) -> Self {
        SipHashState {
            v0: 0x736f6d6570736575 ^ lane_key,
            v1: 0x646f72616e646f6d ^ lane_key,
            v2: 0x6c7967656e657261 ^ lane_key,
            v3: 0x7465646279746573 ^ lane_key,
        }
    }

    #[inline(always)]
    fn update(&mut self, bytes: &[u8; 8]) {
        self.update_u64(u64::from_le_bytes(*bytes));
    }

    #[inline(always)]
    fn update_u64(&mut self, packet: u64) {
        self.v3 ^= packet;
        self.compress::<U>();
        self.v0 ^= packet;
    }

    #[inline(always)]
    fn finalize(mut self) -> u64 {
        self.v2 ^= 0xFF;
        self.compress::<F>();
        (self.v0 ^ self.v1) ^ (self.v2 ^ self.v3)
    }

    #[inline(always)]
    fn compress<const ROUNDS: usize>(&mut self) {
        for _ in 0..ROUNDS {
            self.v0 = self.v0.wrapping_add(self.v1);
            self.v2 = self.v2.wrapping_add(self.v3);
            self.v1 = self.v1.rotate_left(13) ^ self.v0;
            self.v3 = self.v3.rotate_left(16) ^ self.v2;
            self.v0 = self.v0.rotate_left(32);
            self.v2 = self.v2.wrapping_add(self.v1);
            self.v0 = self.v0.wrapping_add(self.v3);
            self.v1 = self.v1.rotate_left(17) ^ self.v2;
            self.v3 = self.v3.rotate_left(21) ^ self.v0;
            self.v2 = self.v2.rotate_left(32);
        }
    }
}

fn compute<const U: usize, const F: usize>(key: [u64; 2], bytes: &[u8]) -> u64 {
    let mut state = SipHashState::<U, F>::new(key);
    let (chunks, rest) = bytes.as_chunks::<8>();
    for chunk in chunks {
        state.update(chunk);
    }
    // SipHash padding: remaining bytes, then zeros, with the total length in
    // the final byte.
    let mut final_packet = [0u8; 8];
    final_packet[..rest.len()].copy_from_slice(rest);
    final_packet[7] = bytes.len() as u8;
    state.update(&final_packet);
    state.finalize()
}

/// SipHash-2-4: fast, cryptographically strong PRF. Matches the reference
/// implementation byte-for-byte.
pub fn sip_hash(key: SipHashKey, bytes: &[u8]) -> u64 {
    compute::<2, 4>(key.0, bytes)
}

/// SipHash-1-3: faster but weaker variant (1 update round, 3 finalization).
pub fn sip_hash13(key: SipHashKey, bytes: &[u8]) -> u64 {
    compute::<1, 3>(key.0, bytes)
}

/// Number of independent SipHash chains interleaved by the tree hash.
pub(crate) const TREE_LANES: usize = 4;

/// The four 8-byte words of a 32-byte packet, little-endian.
fn packet_words(packet: &[u8; 32]) -> [u64; TREE_LANES] {
    core::array::from_fn(|lane| {
        let word: [u8; 8] = core::array::from_fn(|i| packet[lane * 8 + i]);
        u64::from_le_bytes(word)
    })
}

/// The j-lanes tree hash (port of `ScalarSipTreeHashT`): four SipHash
/// chains, one per 8-byte word of each 32-byte packet, reduced to one
/// 64-bit hash by a final SipHash over the four results.
fn tree_compute_scalar<const U: usize, const F: usize>(key: Key, bytes: &[u8]) -> u64 {
    let mut lanes: [SipHashState<U, F>; TREE_LANES] = core::array::from_fn(|lane| {
        // `TREE_LANES | lane` is the lane tag of the reference.
        SipHashState::new_tree_lane(key.0[lane] ^ (TREE_LANES | lane) as u64)
    });
    let (packets, tail) = bytes.as_chunks::<32>();
    for packet in packets {
        for (lane, word) in lanes.iter_mut().zip(packet_words(packet)) {
            lane.update_u64(word);
        }
    }

    for (lane, word) in lanes.iter_mut().zip(packet_words(&final_packet(tail))) {
        lane.update_u64(word);
    }

    // Each lane finalizes; the four results are reduced to one hash.
    let hashes: [u64; TREE_LANES] = lanes.map(|lane| lane.finalize());
    reduce::<U, F>(SipHashKey([key.0[0], key.0[1]]), &hashes)
}

/// The last 32-bit word of the final tree-hash packet: the number of
/// remaining bytes in the top byte, with the last `tail.len() % 4` bytes
/// packed little-endian below it (the HighwayHash padding scheme, so vector
/// and scalar tree hashes agree).
///
/// `tail` is the 0..=31 bytes left after the whole 32-byte packets.
pub(crate) fn final_packet_word(tail: &[u8]) -> u32 {
    let remainder = tail.len();
    debug_assert!(remainder < 32);
    let whole_words = remainder - (remainder & 3);
    let mut packet4 = (remainder as u32) << 24;
    for (i, byte) in tail[whole_words..].iter().enumerate() {
        packet4 += (*byte as u32) << (i * 8);
    }
    packet4
}

/// The final 32-byte packet of the tree hash: the whole 4-byte words of
/// `tail`, zero padding, then [`final_packet_word`] in bytes 28..32.
pub(crate) fn final_packet(tail: &[u8]) -> [u8; 32] {
    let whole_words = tail.len() & !3;
    let mut packet = [0u8; 32];
    packet[..whole_words].copy_from_slice(&tail[..whole_words]);
    packet[28..].copy_from_slice(&final_packet_word(tail).to_le_bytes());
    packet
}

/// The tree hash on the best implementation available: AVX-512VL (with the
/// `avx512` feature) or AVX2 on x86_64 when the CPU has it, NEON on aarch64,
/// the scalar code otherwise. `allow_simd` only ever *disables* SIMD, so
/// callers need no `unsafe`; it is what lets the tests reach both
/// implementations on any one machine.
fn tree_compute<const U: usize, const F: usize>(allow_simd: bool, key: Key, bytes: &[u8]) -> u64 {
    #[cfg(all(
        target_arch = "x86_64",
        feature = "avx512",
        not(feature = "portable-only")
    ))]
    if allow_simd && crate::arch::x86::has_avx512() {
        // SAFETY: AVX-512 support was just detected.
        return unsafe { crate::arch::x86::avx512_sip_tree::tree_hash::<U, F>(key, bytes) };
    }
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
    if allow_simd && crate::arch::x86::has_avx2() {
        // SAFETY: AVX2 support was just detected.
        return unsafe { crate::arch::x86::avx2_sip_tree::tree_hash::<U, F>(key, bytes) };
    }
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        not(feature = "portable-only")
    ))]
    if allow_simd {
        return crate::neon_sip_tree::tree_hash::<U, F>(key, bytes);
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", not(feature = "portable-only")),
        all(
            target_arch = "aarch64",
            target_endian = "little",
            not(feature = "portable-only")
        ),
    )))]
    let _ = allow_simd; // No SIMD tree hash in this build.
    tree_compute_scalar::<U, F>(key, bytes)
}

/// `ReduceSipTreeHash`: SipHash over the lane hashes, with no length
/// padding packet.
pub(crate) fn reduce<const U: usize, const F: usize>(key: SipHashKey, hashes: &[u64]) -> u64 {
    let mut state = SipHashState::<U, F>::new(key.0);
    for hash in hashes {
        state.update_u64(*hash);
    }
    state.finalize()
}

/// SipTreeHash: SipHash-2-4 keyed with 256 bits, hashing 32-byte packets
/// down four interleaved chains (the j-lanes tree hash of the reference;
/// port of `sip_tree_hash.h` / `scalar_sip_tree_hash.cc`). Intended for hash
/// tables holding attacker-controlled data: multi-collisions are
/// infeasible to compute as long as the key remains secret.
///
/// Uses AVX2 on x86_64 CPUs that have it (one 256-bit register holds the four
/// chains; AVX-512VL instead, with the `avx512` feature), NEON on
/// little-endian aarch64 (two 128-bit registers) and the portable code
/// elsewhere; the result is the same.
pub fn sip_tree_hash(key: Key, bytes: &[u8]) -> u64 {
    tree_compute::<2, 4>(true, key, bytes)
}

/// SipTreeHash13: the round-reduced (1 update, 3 finalization) variant of
/// [`sip_tree_hash`].
pub fn sip_tree_hash13(key: Key, bytes: &[u8]) -> u64 {
    tree_compute::<1, 3>(true, key, bytes)
}

/// [`sip_tree_hash`] on the portable code, whatever the CPU. Exposed as
/// `scalar::sip_tree_hash`.
pub fn scalar_sip_tree_hash(key: Key, bytes: &[u8]) -> u64 {
    tree_compute::<2, 4>(false, key, bytes)
}

/// [`sip_tree_hash13`] on the portable code, whatever the CPU. Exposed as
/// `scalar::sip_tree_hash13`.
pub fn scalar_sip_tree_hash13(key: Key, bytes: &[u8]) -> u64 {
    tree_compute::<1, 3>(false, key, bytes)
}

/// Reduces `hashes` to a single 64-bit hash with SipHash-2-4, as
/// [`sip_tree_hash`] combines its four lane results. Useful for tree
/// hashing: hash chunks independently, then reduce their hashes.
pub fn reduce_sip_tree_hash(key: SipHashKey, hashes: &[u64]) -> u64 {
    reduce::<2, 4>(key, hashes)
}

/// See [`reduce_sip_tree_hash`]; the round-reduced variant used by
/// [`sip_tree_hash13`].
pub fn reduce_sip_tree_hash13(key: SipHashKey, hashes: &[u64]) -> u64 {
    reduce::<1, 3>(key, hashes)
}
