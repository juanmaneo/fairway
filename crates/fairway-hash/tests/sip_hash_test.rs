//! Port of upstream `sip_hash_test.cc` (`VerifySipHash`).

mod common;

use fairway_hash::vectors::{
    SIP_HASH_OUTPUT, SIP_KEY, SIP_TREE_HASH13_OUTPUT, SIP_TREE_HASH_OUTPUT, SIP_TREE_KEY,
};
use fairway_hash::{
    reduce_sip_tree_hash, reduce_sip_tree_hash13, scalar, sip_hash, sip_hash13, sip_tree_hash,
    sip_tree_hash13, Key, SipHash13State, SipHashKey, SipHashState,
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

/// The public incremental state (upstream `SipHashState`): `update_bytes`
/// reproduces `sip_hash`, and feeding packets by hand with the padding
/// packet built by the caller gives the same hash.
#[test]
fn sip_hash_state_matches_sip_hash() {
    let input = common::golden_input();
    let key = SipHashKey::from(SIP_KEY);
    for size in 0..64 {
        let bytes = &input[..size];
        let mut state = SipHashState::new(key);
        state.update_bytes(bytes);
        assert_eq!(state.finalize(), SIP_HASH_OUTPUT[size], "length {size}");

        let mut state = SipHash13State::new(key);
        state.update_bytes(bytes);
        assert_eq!(
            state.finalize(),
            sip_hash13(key, bytes),
            "1-3, length {size}"
        );

        let mut state = SipHashState::new(key);
        let (packets, rest) = bytes.as_chunks::<8>();
        for packet in packets {
            state.update(packet);
        }
        let mut last = [0u8; 8];
        last[..rest.len()].copy_from_slice(rest);
        last[7] = size as u8;
        state.update(&last);
        assert_eq!(
            state.finalize(),
            SIP_HASH_OUTPUT[size],
            "by hand, length {size}"
        );
    }
}

/// A cloned state continues independently of the original.
#[test]
fn sip_hash_state_clone_forks() {
    let key = SipHashKey::from(SIP_KEY);
    let mut state = SipHashState::new(key);
    state.update(&[1; 8]);
    let fork = state.clone();
    state.update(&[2; 8]);
    let mut expected = SipHashState::new(key);
    expected.update(&[1; 8]);
    assert_eq!(fork.finalize(), expected.finalize());
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

/// Checks a (2-4, 1-3) pair of tree hash functions against golden values
/// generated with the reference `ScalarSipTreeHash` / `ScalarSipTreeHash13`
/// (upstream has none) for inputs `0, 1, 2, ...`.
fn check_tree_hash(tree24: fn(Key, &[u8]) -> u64, tree13: fn(Key, &[u8]) -> u64) {
    let mut input = vec![0u8; 65536];
    for (i, byte) in input.iter_mut().enumerate() {
        *byte = i as u8;
    }
    let key = Key(SIP_TREE_KEY);
    for (size, expected) in SIP_TREE_HASH_OUTPUT {
        assert_eq!(
            tree24(key, &input[..size]),
            expected,
            "tree24 mismatch at length {size}"
        );
    }
    for (size, expected) in SIP_TREE_HASH13_OUTPUT {
        assert_eq!(
            tree13(key, &input[..size]),
            expected,
            "tree13 mismatch at length {size}"
        );
    }
    // The two variants are distinct functions.
    assert_ne!(tree24(key, &input[..100]), tree13(key, &input[..100]));
}

/// The best implementation for this CPU (AVX2 on x86_64 machines that have it).
#[test]
fn verify_sip_tree_hash() {
    check_tree_hash(sip_tree_hash, sip_tree_hash13);
}

/// The portable implementation, which `sip_tree_hash` falls back to.
#[test]
fn verify_scalar_sip_tree_hash() {
    check_tree_hash(scalar::sip_tree_hash, scalar::sip_tree_hash13);
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
