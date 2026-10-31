//! Deterministic stand-ins for upstream `highwayhash_fuzzer.cc` and
//! `sip_hash_fuzzer.cc`.
//!
//! The libFuzzer targets take the key from the first bytes of the input and
//! hash the rest, only checking nothing crashes. Here the same shape is driven
//! by a fixed-seed generator (so CI is reproducible), and, since every
//! backend must produce identical output, each result is also cross-checked
//! against the portable implementation, over random splits.

mod common;

use common::{backends, Rng};
use fairway::{scalar, sip_hash, sip_hash13, HighwayHasher, Key, SipHashKey};

const CASES: usize = 2_000;
const MAX_LEN: usize = 300;

/// Splits `data` at random points and streams it into `h`.
fn update_in_random_pieces(h: &mut HighwayHasher, data: &[u8], rng: &mut Rng) {
    let mut rest = data;
    while !rest.is_empty() {
        let n = rng.below(rest.len() + 1);
        h.update(&rest[..n]);
        rest = &rest[n..];
    }
}

#[test]
fn highwayhash_fuzzer() {
    let backends = backends();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for case in 0..CASES {
        let len = 32 + rng.below(MAX_LEN);
        let fuzz_input = rng.bytes(len);
        let (key_bytes, data) = fuzz_input.split_at(32);
        let key = Key(core::array::from_fn(|i| {
            u64::from_le_bytes(key_bytes[i * 8..i * 8 + 8].try_into().unwrap())
        }));

        let expected = (
            scalar::hash64(key, data),
            scalar::hash128(key, data),
            scalar::hash256(key, data),
        );
        for &backend in &backends {
            let mut h = HighwayHasher::with_backend(key, backend);
            update_in_random_pieces(&mut h, data, &mut rng);
            let got = (h.finalize64(), h.finalize128(), h.finalize256());
            assert_eq!(
                got,
                expected,
                "case {case}: {backend:?} != portable, len {}",
                data.len()
            );
        }
    }
}

#[test]
fn sip_hash_fuzzer() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for _ in 0..CASES {
        let len = 16 + rng.below(MAX_LEN);
        let fuzz_input = rng.bytes(len);
        let (key_bytes, data) = fuzz_input.split_at(16);
        let key = SipHashKey(core::array::from_fn(|i| {
            u64::from_le_bytes(key_bytes[i * 8..i * 8 + 8].try_into().unwrap())
        }));
        assert_eq!(sip_hash(key, data), sip_hash(key, data));
        assert_eq!(sip_hash13(key, data), sip_hash13(key, data));
    }
}
