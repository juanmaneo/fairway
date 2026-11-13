//! Port of upstream `highwayhash_fuzzer.cc`, which only checks that nothing
//! crashes. Here every backend must also agree with the portable code, however
//! the input is streamed.

#![no_main]

use fairway_hash::{fairway128, fairway256, fairway64, scalar, HighwayHasher, Key};
use fairway_hash_fuzz::{available_backends, exact};
use libfuzzer_sys::arbitrary::{self, Arbitrary};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    key: [u64; 4],
    data: Vec<u8>,
    /// Sizes of the pieces `data` is streamed in (what is left after the last
    /// one goes in as a final piece).
    pieces: Vec<u8>,
    /// Where `data` is split between `update` and the keyed `hash*` calls.
    split: u16,
}

fuzz_target!(|input: Input| {
    let key = Key(input.key);
    let data = exact(&input.data);
    let want = (
        scalar::hash64(key, &data),
        scalar::hash128(key, &data),
        scalar::hash256(key, &data),
    );

    assert_eq!(
        (
            fairway64(key, &data),
            fairway128(key, &data),
            fairway256(key, &data)
        ),
        want,
        "one-shot, len {}",
        data.len()
    );

    for &backend in available_backends() {
        let context = || format!("{backend:?}, len {}", data.len());

        // Streamed in arbitrary pieces. Finalizing must not change the state.
        let mut hasher = HighwayHasher::with_backend(key, backend);
        let mut rest: &[u8] = &data;
        for &size in &input.pieces {
            let (piece, tail) = rest.split_at((size as usize).min(rest.len()));
            hasher.update(piece);
            rest = tail;
        }
        hasher.update(rest);
        for _ in 0..2 {
            let got = (
                hasher.finalize64(),
                hasher.finalize128(),
                hasher.finalize256(),
            );
            assert_eq!(got, want, "streamed, {}", context());
        }

        // The keyed one-shot calls: from a fresh hasher, whose tail is read
        // straight from `data`, and after `split` bytes were streamed in.
        let fresh = HighwayHasher::with_backend(key, backend);
        let got = (
            fresh.hash64(&data),
            fresh.hash128(&data),
            fresh.hash256(&data),
        );
        assert_eq!(got, want, "keyed, {}", context());

        let split = input.split as usize % (data.len() + 1);
        let mut hasher = HighwayHasher::with_backend(key, backend);
        hasher.update(&data[..split]);
        let got = (
            hasher.hash64(&data[split..]),
            hasher.hash128(&data[split..]),
            hasher.hash256(&data[split..]),
        );
        assert_eq!(got, want, "keyed after {split} bytes, {}", context());
        // `hash*` leaves the hasher as it was.
        assert_eq!(
            hasher.finalize64(),
            scalar::hash64(key, &data[..split]),
            "hash64 changed the hasher, {}",
            context()
        );

        // `reset` starts over, on the same backend.
        hasher.reset(key);
        hasher.update(&data);
        assert_eq!(hasher.finalize256(), want.2, "reset, {}", context());
    }
});
