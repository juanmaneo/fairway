//! Port of upstream `sip_hash_test.cc` (`VerifySipHash`).

mod common;

use fairway_hash::vectors::{
    SIP_HASH_OUTPUT, SIP_KEY, SIP_TREE_HASH13_OUTPUT, SIP_TREE_HASH_OUTPUT, SIP_TREE_KEY,
};
use fairway_hash::{
    reduce_sip_tree_hash, reduce_sip_tree_hash13, sip_hash, sip_hash13, sip_tree_hash,
    sip_tree_hash13, Key, SipHashKey,
};

/// Known-good SipHash-2-4 output from D. Bernstein, for the empty string,
/// `00`, `00 01`, ... up to 63 bytes.
#[test]
fn verify_sip_hash() {
    let input = common::golden_input();
    let key = SipHashKey::from(SIP_KEY);
    for size in 0..64 {
        assert_eq!(
            sip_hash(key, &input[..size]),
            SIP_HASH_OUTPUT[size],
            "Mismatch at length {size}"
        );
    }
}

/// Rust-specific: upstream has no SipHash-1-3 vectors, so check it is a
/// distinct, deterministic, length- and order-sensitive function.
#[test]
fn sip_hash13_sanity() {
    let key = SipHashKey(SIP_KEY);
    assert_eq!(sip_hash13(key, b""), sip_hash13(key, b""));
    assert_ne!(sip_hash13(key, b""), sip_hash13(key, b"\0"));
    assert_ne!(sip_hash13(key, b"ab"), sip_hash13(key, b"ba"));
    assert_ne!(sip_hash13(key, b"message"), sip_hash(key, b"message"));
}

/// Golden values generated with the reference `ScalarSipTreeHash` /
/// `ScalarSipTreeHash13` (upstream has none) for inputs `0, 1, 2, ...`.
#[test]
fn verify_sip_tree_hash() {
    let mut input = vec![0u8; 65536];
    for (i, byte) in input.iter_mut().enumerate() {
        *byte = i as u8;
    }
    let key = Key(SIP_TREE_KEY);
    for (size, expected) in SIP_TREE_HASH_OUTPUT {
        assert_eq!(
            sip_tree_hash(key, &input[..size]),
            expected,
            "tree24 mismatch at length {size}"
        );
    }
    for (size, expected) in SIP_TREE_HASH13_OUTPUT {
        assert_eq!(
            sip_tree_hash13(key, &input[..size]),
            expected,
            "tree13 mismatch at length {size}"
        );
    }
    // The two variants are distinct functions.
    assert_ne!(
        sip_tree_hash(key, &input[..100]),
        sip_tree_hash13(key, &input[..100])
    );
}

/// The reduce step matches the reference `ReduceSipTreeHash`, and the
/// public reduction used by `sip_tree_hash` is the 2-4 one.
#[test]
fn verify_reduce_sip_tree_hash() {
    let key = SipHashKey([SIP_TREE_KEY[0], SIP_TREE_KEY[1]]);
    let hashes = [1, 2, 3, 4];
    assert_eq!(reduce_sip_tree_hash(key, &hashes), 0x6666b9ee4bb82521);
    assert_eq!(reduce_sip_tree_hash13(key, &hashes), 0xcd1eaa66384a8a0d);
    let hashes = [0x0123456789abcdef, 0, u64::MAX, 0xdeadbeefdeadbeef];
    assert_eq!(reduce_sip_tree_hash(key, &hashes), 0x06ffd356f916f3a9);
    // Reduction is order-sensitive and length-sensitive.
    assert_ne!(
        reduce_sip_tree_hash(key, &hashes),
        reduce_sip_tree_hash(key, &hashes.iter().rev().copied().collect::<Vec<_>>())
    );
    assert_ne!(
        reduce_sip_tree_hash(key, &hashes[..3]),
        reduce_sip_tree_hash(key, &hashes)
    );
}
