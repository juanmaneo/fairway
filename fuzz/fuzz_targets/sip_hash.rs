//! Port of upstream `sip_hash_fuzzer.cc`, with the hashes checked against
//! independent implementations: the standard library's SipHash-2-4 and the
//! specification-based reference in this crate (also for SipHash-1-3, which
//! the standard library doesn't expose).

#![no_main]
#![allow(deprecated)] // `SipHasher` is deprecated, but it is the reference.

use std::hash::{Hasher, SipHasher};

use fairway_hash::{sip_hash, sip_hash13, SipHashKey};
use fairway_hash_fuzz::{exact, reference};
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    key: [u64; 2],
    data: Vec<u8>,
}

fuzz_target!(|input: Input| {
    let key = SipHashKey(input.key);
    let data = exact(&input.data);

    let mut std_sip = SipHasher::new_with_keys(input.key[0], input.key[1]);
    std_sip.write(&data);
    let want24 = std_sip.finish();
    assert_eq!(
        reference::siphash(2, 4, input.key, &data),
        want24,
        "the reference disagrees with std, len {}",
        data.len()
    );
    assert_eq!(
        sip_hash(key, &data),
        want24,
        "SipHash-2-4, len {}",
        data.len()
    );

    assert_eq!(
        sip_hash13(key, &data),
        reference::siphash(1, 3, input.key, &data),
        "SipHash-1-3, len {}",
        data.len()
    );
});
