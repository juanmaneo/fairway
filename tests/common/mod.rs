//! Helpers shared by the ports of the upstream C++ tests.

#![allow(dead_code)] // Each test binary uses a different subset.

use fairway_hash::{Backend, HighwayHasher, Key};

/// Every backend, in the order the C++ `InstructionSets::RunAll` visits them.
pub const ALL_BACKENDS: [Backend; 4] = [
    Backend::Portable,
    Backend::Sse41,
    Backend::Avx2,
    Backend::Neon,
];

/// The backends this CPU/build can actually run (`with_backend` falls back to
/// portable for the others). Printed so a CI log shows what was covered.
pub fn backends() -> Vec<Backend> {
    let available: Vec<Backend> = ALL_BACKENDS
        .into_iter()
        .filter(|&b| HighwayHasher::with_backend(Key([0; 4]), b).backend() == b)
        .collect();
    eprintln!("backends under test: {available:?}");
    available
}

/// `char in[kMaxSize + 1]`, where `in[size] = size`: hashing `&in[..size]`
/// gives the empty string, then `00`, `00 01`, ...
pub fn golden_input() -> Vec<u8> {
    (0..=64).collect()
}

/// Deterministic xorshift64* generator for the fuzz-style tests (no deps).
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64() as u8).collect()
    }
}
