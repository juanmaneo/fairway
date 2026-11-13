//! The batch API (`fairway*_many`, `fairway*_many_par`) against the upstream
//! golden vectors and the portable reference implementation.

mod common;

use common::{golden_input, Rng};
use fairway_hash::vectors::{EXPECTED128, EXPECTED256, EXPECTED64, KEY64};
use fairway_hash::{fairway128_many, fairway256_many, fairway64_many, scalar, Key};
#[cfg(feature = "std")]
use fairway_hash::{fairway128_many_par, fairway256_many_par, fairway64_many_par};

/// All 65 golden inputs (sizes 0..=64) as one batch: different lengths in the
/// same group, leftover inputs after the last full group, and empty inputs.
fn golden_batch() -> Vec<Vec<u8>> {
    let input = golden_input();
    (0..=64).map(|size| input[..size].to_vec()).collect()
}

#[test]
fn many_matches_golden_vectors() {
    let key = Key(KEY64);
    let batch = golden_batch();

    let mut out64 = vec![0u64; batch.len()];
    fairway64_many(key, &batch, &mut out64);
    assert_eq!(out64, EXPECTED64);

    let mut out128 = vec![[0u64; 2]; batch.len()];
    fairway128_many(key, &batch, &mut out128);
    assert_eq!(out128, EXPECTED128);

    let mut out256 = vec![[0u64; 4]; batch.len()];
    fairway256_many(key, &batch, &mut out256);
    assert_eq!(out256, EXPECTED256);
}

#[cfg(feature = "std")]
#[test]
fn many_par_matches_golden_vectors() {
    let key = Key(KEY64);
    let batch = golden_batch();

    let mut out64 = vec![0u64; batch.len()];
    fairway64_many_par(key, &batch, &mut out64);
    assert_eq!(out64, EXPECTED64);

    let mut out128 = vec![[0u64; 2]; batch.len()];
    fairway128_many_par(key, &batch, &mut out128);
    assert_eq!(out128, EXPECTED128);

    let mut out256 = vec![[0u64; 4]; batch.len()];
    fairway256_many_par(key, &batch, &mut out256);
    assert_eq!(out256, EXPECTED256);
}

/// Random inputs, long enough that groups are advanced in lockstep over many
/// packets (the golden inputs are at most two packets long), with a mix of
/// equal and very different lengths.
fn random_batch(rng: &mut Rng, inputs: usize, max_len: usize) -> Vec<Vec<u8>> {
    let mut batch: Vec<Vec<u8>> = (0..inputs)
        .map(|_| {
            let len = rng.below(max_len + 1);
            rng.bytes(len)
        })
        .collect();
    // A run of equal lengths, so some groups have nothing left over after the
    // common prefix.
    let len = rng.below(max_len + 1);
    batch.extend((0..8).map(|_| rng.bytes(len)));
    batch
}

#[test]
fn many_matches_portable_on_long_inputs() {
    let key = Key([0xdead_beef, 0x1234_5678_9abc_def0, 7, u64::MAX]);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for (inputs, max_len) in [(13, 100), (37, 2_000), (23, 10_000)] {
        let batch = random_batch(&mut rng, inputs, max_len);

        let mut out64 = vec![0u64; batch.len()];
        fairway64_many(key, &batch, &mut out64);
        let mut out128 = vec![[0u64; 2]; batch.len()];
        fairway128_many(key, &batch, &mut out128);
        let mut out256 = vec![[0u64; 4]; batch.len()];
        fairway256_many(key, &batch, &mut out256);

        for (i, input) in batch.iter().enumerate() {
            let ctx = format!("input {i} of {} ({} bytes)", batch.len(), input.len());
            assert_eq!(out64[i], scalar::hash64(key, input), "64: {ctx}");
            assert_eq!(out128[i], scalar::hash128(key, input), "128: {ctx}");
            assert_eq!(out256[i], scalar::hash256(key, input), "256: {ctx}");
        }
    }
}

/// Enough data (about 8 MiB) for the parallel functions to really use several
/// threads on a multi-core machine.
#[cfg(feature = "std")]
#[test]
fn many_par_matches_many_on_large_batches() {
    let key = Key([1, 2, 3, 4]);
    let mut rng = Rng(42);
    let batch = random_batch(&mut rng, 200, 80_000);

    let mut expected64 = vec![0u64; batch.len()];
    fairway64_many(key, &batch, &mut expected64);
    let mut par64 = vec![0u64; batch.len()];
    fairway64_many_par(key, &batch, &mut par64);
    assert_eq!(par64, expected64);

    let mut expected128 = vec![[0u64; 2]; batch.len()];
    fairway128_many(key, &batch, &mut expected128);
    let mut par128 = vec![[0u64; 2]; batch.len()];
    fairway128_many_par(key, &batch, &mut par128);
    assert_eq!(par128, expected128);

    let mut expected256 = vec![[0u64; 4]; batch.len()];
    fairway256_many(key, &batch, &mut expected256);
    let mut par256 = vec![[0u64; 4]; batch.len()];
    fairway256_many_par(key, &batch, &mut par256);
    assert_eq!(par256, expected256);
}

#[test]
fn empty_batch_is_fine() {
    let inputs: [&[u8]; 0] = [];
    let mut out: [u64; 0] = [];
    fairway64_many(Key(KEY64), &inputs, &mut out);
}
