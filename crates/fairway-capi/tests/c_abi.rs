//! The exported C functions against the upstream golden vectors, called the
//! way a C program would (raw pointers, `size` as `u64`).

use core::ffi::c_char;

use fairway_capi::*;
use fairway_hash::vectors::{EXPECTED64, KEY64, SIP_HASH_OUTPUT, SIP_KEY};
use fairway_hash::{sip_hash13, SipHashKey};

type HighwayFn = unsafe extern "C" fn(*const u64, *const c_char, u64) -> u64;

const HIGHWAY_FNS: [(&str, HighwayFn); 6] = [
    ("HighwayHash64", HighwayHash64),
    ("Portable", HighwayHash64_TargetPortable),
    ("SSE41", HighwayHash64_TargetSSE41),
    ("AVX2", HighwayHash64_TargetAVX2),
    ("NEON", HighwayHash64_TargetNEON),
    ("VSX", HighwayHash64_TargetVSX),
];

fn golden_input() -> Vec<u8> {
    (0..=64).collect()
}

#[test]
fn highway_hash64_matches_the_golden_vectors_on_every_target() {
    let input = golden_input();
    for (name, f) in HIGHWAY_FNS {
        for (size, expected) in EXPECTED64.iter().enumerate() {
            // SAFETY: `KEY64` has 4 words and `input` has more than `size` bytes.
            let hash = unsafe { f(KEY64.as_ptr(), input.as_ptr().cast(), size as u64) };
            assert_eq!(hash, *expected, "{name} at length {size}");
        }
    }
}

#[test]
fn sip_hash_matches_the_reference_vectors() {
    let input = golden_input();
    for size in 0..64 {
        // SAFETY: `SIP_KEY` has 2 words and `input` has more than `size` bytes.
        let hash = unsafe { SipHashC(SIP_KEY.as_ptr(), input.as_ptr().cast(), size as u64) };
        assert_eq!(hash, SIP_HASH_OUTPUT[size], "SipHashC at length {size}");
        // SAFETY: as above.
        let hash13 = unsafe { SipHash13C(SIP_KEY.as_ptr(), input.as_ptr().cast(), size as u64) };
        assert_eq!(hash13, sip_hash13(SipHashKey(SIP_KEY), &input[..size]));
    }
}

#[test]
fn keys_need_no_alignment() {
    let input = golden_input();
    // A key 1 byte into a buffer is misaligned for u64.
    let mut bytes = [0u8; 1 + 32];
    let (chunks, _) = bytes[1..].as_chunks_mut::<8>();
    for (word, chunk) in KEY64.iter().zip(chunks) {
        *chunk = word.to_ne_bytes();
    }
    // SAFETY: 32 key bytes follow offset 1 and `input` has 64 bytes.
    let hash = unsafe { HighwayHash64(bytes[1..].as_ptr().cast(), input.as_ptr().cast(), 64) };
    assert_eq!(hash, EXPECTED64[64]);
}

#[test]
fn empty_input_may_be_a_null_pointer() {
    for (name, f) in HIGHWAY_FNS {
        // SAFETY: `size` is 0, so the null pointer is never read.
        let hash = unsafe { f(KEY64.as_ptr(), core::ptr::null(), 0) };
        assert_eq!(hash, EXPECTED64[0], "{name}");
    }
    // SAFETY: as above.
    let sip = unsafe { SipHashC(SIP_KEY.as_ptr(), core::ptr::null(), 0) };
    assert_eq!(sip, SIP_HASH_OUTPUT[0]);
}
