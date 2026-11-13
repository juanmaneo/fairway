//! `fairway{64,128,256}_many` and their `_many_par` variants must return, for
//! every input, what hashing it alone returns. The batch code interleaves
//! several inputs once they share enough whole packets, so the inputs here
//! are long and of uneven lengths, which a fuzzer would rarely produce from
//! raw bytes alone: each is a window of one shared pool of bytes.

#![no_main]

use fairway_hash::{
    fairway128_many, fairway128_many_par, fairway256_many, fairway256_many_par, fairway64_many,
    fairway64_many_par, scalar, Key,
};
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;

/// Longest single input, enough for several interleaved groups of packets.
const MAX_LEN: usize = 4096;
/// Most inputs per batch: two full groups of six, plus a partial group.
const MAX_INPUTS: usize = 14;

#[derive(Arbitrary, Debug)]
struct Input {
    key: [u64; 4],
    /// The bytes the inputs are cut from, repeated as often as needed.
    pool: Vec<u8>,
    /// `(start, length)` of each input in the pool.
    windows: Vec<(u16, u16)>,
}

/// `len` bytes of `pool` from `start`, wrapping around, in an exact allocation.
fn window(pool: &[u8], start: usize, len: usize) -> Box<[u8]> {
    if pool.is_empty() {
        return vec![0; len].into_boxed_slice();
    }
    (0..len)
        .map(|i| pool[(start + i) % pool.len()])
        .collect::<Vec<u8>>()
        .into_boxed_slice()
}

fuzz_target!(|input: Input| {
    let key = Key(input.key);
    let inputs: Vec<Box<[u8]>> = input
        .windows
        .iter()
        .take(MAX_INPUTS)
        .map(|&(start, len)| window(&input.pool, start as usize, len as usize % (MAX_LEN + 1)))
        .collect();

    let want64: Vec<u64> = inputs.iter().map(|i| scalar::hash64(key, i)).collect();
    let want128: Vec<[u64; 2]> = inputs.iter().map(|i| scalar::hash128(key, i)).collect();
    let want256: Vec<[u64; 4]> = inputs.iter().map(|i| scalar::hash256(key, i)).collect();
    let lens = || inputs.iter().map(|i| i.len()).collect::<Vec<_>>();

    let mut got = vec![0u64; inputs.len()];
    fairway64_many(key, &inputs, &mut got);
    assert_eq!(got, want64, "fairway64_many, lengths {:?}", lens());
    got.fill(0);
    fairway64_many_par(key, &inputs, &mut got);
    assert_eq!(got, want64, "fairway64_many_par, lengths {:?}", lens());

    let mut got = vec![[0u64; 2]; inputs.len()];
    fairway128_many(key, &inputs, &mut got);
    assert_eq!(got, want128, "fairway128_many, lengths {:?}", lens());
    got.fill([0; 2]);
    fairway128_many_par(key, &inputs, &mut got);
    assert_eq!(got, want128, "fairway128_many_par, lengths {:?}", lens());

    let mut got = vec![[0u64; 4]; inputs.len()];
    fairway256_many(key, &inputs, &mut got);
    assert_eq!(got, want256, "fairway256_many, lengths {:?}", lens());
    got.fill([0; 4]);
    fairway256_many_par(key, &inputs, &mut got);
    assert_eq!(got, want256, "fairway256_many_par, lengths {:?}", lens());
});
