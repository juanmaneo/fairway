//! The SIMD tree hash (AVX2, AVX-512VL with the `avx512` feature, NEON) must
//! equal the portable one for every input length, key and alignment. Upstream
//! has no fuzzer for it. `reduce_sip_tree_hash` is checked against the
//! reference SipHash.

#![no_main]

use fairway_hash::{
    reduce_sip_tree_hash, reduce_sip_tree_hash13, scalar, sip_tree_hash, sip_tree_hash13, Key,
    SipHashKey,
};
use fairway_hash_fuzz::{exact, reference};
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    key: [u64; 4],
    data: Vec<u8>,
    /// How far into its allocation the hashed bytes start (mod 32), to
    /// exercise unaligned loads.
    offset: u8,
    reduce_key: [u64; 2],
    hashes: Vec<u64>,
}

fn check_tree(key: Key, bytes: &[u8], what: &str) {
    assert_eq!(
        sip_tree_hash(key, bytes),
        scalar::sip_tree_hash(key, bytes),
        "tree24, {what}, len {}",
        bytes.len()
    );
    assert_eq!(
        sip_tree_hash13(key, bytes),
        scalar::sip_tree_hash13(key, bytes),
        "tree13, {what}, len {}",
        bytes.len()
    );
}

fuzz_target!(|input: Input| {
    let key = Key(input.key);

    // Flush with the end of an exact allocation, so an over-read is reported.
    let data = exact(&input.data);
    check_tree(key, &data, "aligned");

    // The same bytes starting `offset` bytes into a larger allocation.
    let offset = (input.offset % 32) as usize;
    let mut shifted = vec![0u8; offset];
    shifted.extend_from_slice(&input.data);
    let shifted = exact(&shifted);
    check_tree(key, &shifted[offset..], "unaligned");

    let reduce_key = SipHashKey(input.reduce_key);
    assert_eq!(
        reduce_sip_tree_hash(reduce_key, &input.hashes),
        reference::reduce(2, 4, input.reduce_key, &input.hashes),
        "reduce24, {} hashes",
        input.hashes.len()
    );
    assert_eq!(
        reduce_sip_tree_hash13(reduce_key, &input.hashes),
        reference::reduce(1, 3, input.reduce_key, &input.hashes),
        "reduce13, {} hashes",
        input.hashes.len()
    );
});
