//! Portable SipHash-2-4 and SipHash-1-3 (port of `sip_hash.h`).
//!
//! Outputs are identical to the reference SipHash for messages of any length.

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

    #[inline(always)]
    fn update(&mut self, bytes: &[u8; 8]) {
        let packet = u64::from_le_bytes(*bytes);
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
