//! Port of upstream `sip_hash_test.cc` (`VerifySipHash`).

mod common;

use fairway::vectors::{SIP_HASH_OUTPUT, SIP_KEY};
use fairway::{sip_hash, sip_hash13, SipHashKey};

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
