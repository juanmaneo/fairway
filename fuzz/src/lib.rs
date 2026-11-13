//! Helpers shared by the fuzz targets.
//!
//! The targets are differential: each hashes the same input through several
//! implementations that must agree (every SIMD backend against the portable
//! one, the batch functions against one-at-a-time hashing, SipHash against
//! an independent reference), so a failure means a real bug without needing
//! a known-good hash value. They also run under AddressSanitizer, which is
//! what checks the `unsafe` SIMD loads stay inside their input; see
//! [`exact`].

use std::sync::OnceLock;

use fairway_hash::{Backend, HighwayHasher, Key};

/// The HighwayHash backends this CPU and build can run: the portable one
/// plus whichever of SSE4.1, AVX2, AVX-512VL (with the `avx512` feature) and
/// NEON are available. `with_backend` falls back to portable for the rest,
/// so those are filtered out rather than compared with themselves.
pub fn available_backends() -> &'static [Backend] {
    static BACKENDS: OnceLock<Vec<Backend>> = OnceLock::new();
    BACKENDS.get_or_init(|| {
        [
            Backend::Portable,
            Backend::Sse41,
            Backend::Avx2,
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            Backend::Avx512,
            Backend::Neon,
        ]
        .into_iter()
        .filter(|&b| HighwayHasher::with_backend(Key([0; 4]), b).backend() == b)
        .collect()
    })
}

/// A copy of `bytes` in an allocation of exactly that size.
///
/// A `Vec` built by `Arbitrary` usually has spare capacity, and a SIMD load
/// that ran past the end of the data would read it without AddressSanitizer
/// noticing. An exact allocation puts the end of the input on the end of the
/// heap block, so any such over-read is reported.
pub fn exact(bytes: &[u8]) -> Box<[u8]> {
    bytes.into()
}

/// SipHash written from the specification, with no code shared with the
/// crate under test, so it can referee the crate's SipHash and tree hash
/// reduction (the one-shot SipHash has a single implementation, so there is
/// nothing else to compare it with).
pub mod reference {
    /// A SipHash state with `c` compression rounds per word and `d`
    /// finalization rounds.
    pub struct Sip {
        v: [u64; 4],
        c: usize,
        d: usize,
    }

    impl Sip {
        pub fn new(c: usize, d: usize, key: [u64; 2]) -> Self {
            let [k0, k1] = key;
            Sip {
                v: [
                    k0 ^ 0x736f_6d65_7073_6575,
                    k1 ^ 0x646f_7261_6e64_6f6d,
                    k0 ^ 0x6c79_6765_6e65_7261,
                    k1 ^ 0x7465_6462_7974_6573,
                ],
                c,
                d,
            }
        }

        fn round(&mut self) {
            let v = &mut self.v;
            v[0] = v[0].wrapping_add(v[1]);
            v[1] = v[1].rotate_left(13);
            v[1] ^= v[0];
            v[0] = v[0].rotate_left(32);
            v[2] = v[2].wrapping_add(v[3]);
            v[3] = v[3].rotate_left(16);
            v[3] ^= v[2];
            v[0] = v[0].wrapping_add(v[3]);
            v[3] = v[3].rotate_left(21);
            v[3] ^= v[0];
            v[2] = v[2].wrapping_add(v[1]);
            v[1] = v[1].rotate_left(17);
            v[1] ^= v[2];
            v[2] = v[2].rotate_left(32);
        }

        pub fn word(&mut self, m: u64) {
            self.v[3] ^= m;
            for _ in 0..self.c {
                self.round();
            }
            self.v[0] ^= m;
        }

        pub fn finish(mut self) -> u64 {
            self.v[2] ^= 0xff;
            for _ in 0..self.d {
                self.round();
            }
            self.v[0] ^ self.v[1] ^ self.v[2] ^ self.v[3]
        }
    }

    /// SipHash-`c`-`d` of `data`: whole little-endian words, then the last
    /// 0..=7 bytes with the length (mod 256) in the top byte.
    pub fn siphash(c: usize, d: usize, key: [u64; 2], data: &[u8]) -> u64 {
        let mut sip = Sip::new(c, d, key);
        let (words, rest) = data.as_chunks::<8>();
        for word in words {
            sip.word(u64::from_le_bytes(*word));
        }
        let mut last = [0u8; 8];
        last[..rest.len()].copy_from_slice(rest);
        sip.word(u64::from_le_bytes(last) | (data.len() as u64) << 56);
        sip.finish()
    }

    /// What `reduce_sip_tree_hash` computes: SipHash over the 64-bit
    /// `hashes` with no length word.
    pub fn reduce(c: usize, d: usize, key: [u64; 2], hashes: &[u64]) -> u64 {
        let mut sip = Sip::new(c, d, key);
        for &hash in hashes {
            sip.word(hash);
        }
        sip.finish()
    }
}
