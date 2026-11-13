//! Port of upstream `highwayhash_test.cc` + `highwayhash_test_target.cc`:
//! `VerifyImplementations` (golden vectors on every target) and `VerifyCat`
//! (`HighwayHashCatT` streaming equals one-shot hashing on every target).

mod common;

use common::{backends, golden_input};
use fairway_hash::vectors::{EXPECTED128, EXPECTED256, EXPECTED64, KEY64};
use fairway_hash::{fairway128, fairway256, fairway64, scalar, Backend, HighwayHasher, Key};

/// The three `HHResult64/128/256` instantiations of the C++ templates.
#[derive(Clone, Copy, Debug)]
enum Width {
    W64,
    W128,
    W256,
}

const WIDTHS: [Width; 3] = [Width::W64, Width::W128, Width::W256];

fn finalize(h: &HighwayHasher, width: Width) -> Vec<u64> {
    match width {
        Width::W64 => vec![h.finalize64()],
        Width::W128 => h.finalize128().to_vec(),
        Width::W256 => h.finalize256().to_vec(),
    }
}

fn hash(backend: Backend, key: Key, bytes: &[u8], width: Width) -> Vec<u64> {
    let mut h = HighwayHasher::with_backend(key, backend);
    h.update(bytes);
    finalize(&h, width)
}

fn known_good(width: Width, size: usize) -> Vec<u64> {
    match width {
        Width::W64 => vec![EXPECTED64[size]],
        Width::W128 => EXPECTED128[size].to_vec(),
        Width::W256 => EXPECTED256[size].to_vec(),
    }
}

/// `VerifyImplementations<Result>`: `InstructionSets::RunAll<HighwayHashTest>`.
fn verify_implementations(width: Width) {
    let key = Key(KEY64);
    let input = golden_input();
    for backend in backends() {
        for size in 0..=64 {
            assert_eq!(
                hash(backend, key, &input[..size], width),
                known_good(width, size),
                "Mismatch at size {size} for target {} ({width:?})",
                backend.name()
            );
        }
    }
}

#[test]
fn verify_implementations_64() {
    verify_implementations(Width::W64);
}

#[test]
fn verify_implementations_128() {
    verify_implementations(Width::W128);
}

#[test]
fn verify_implementations_256() {
    verify_implementations(Width::W256);
}

/// `InstructionSets::Run<HighwayHash>` (the best target) and the portable
/// target called directly, through the one-shot API.
#[test]
fn one_shot_functions_match_golden() {
    let key = Key(KEY64);
    let input = golden_input();
    for size in 0..=64 {
        let bytes = &input[..size];
        assert_eq!(fairway64(key, bytes), EXPECTED64[size], "best 64 at {size}");
        assert_eq!(
            fairway128(key, bytes),
            EXPECTED128[size],
            "best 128 at {size}"
        );
        assert_eq!(
            fairway256(key, bytes),
            EXPECTED256[size],
            "best 256 at {size}"
        );
        assert_eq!(
            scalar::hash64(key, bytes),
            EXPECTED64[size],
            "scalar 64 at {size}"
        );
        assert_eq!(
            scalar::hash128(key, bytes),
            EXPECTED128[size],
            "scalar 128 at {size}"
        );
        assert_eq!(
            scalar::hash256(key, bytes),
            EXPECTED256[size],
            "scalar 256 at {size}"
        );
    }
}

/// `HighwayHasher::new` picks the best backend, and the one-shot helpers
/// (which feed whole packets straight from the input) agree with the
/// incremental hasher on inputs of many packets, with and without a tail.
#[test]
fn new_and_one_shot_agree_on_long_inputs() {
    let key = Key(KEY64);
    let input = golden_input();
    let mut h = HighwayHasher::new(key);
    assert_eq!(h.backend(), fairway_hash::best_backend());
    h.update(&input[..64]);
    assert_eq!(h.finalize64(), EXPECTED64[64]);

    let mut rng = common::Rng(7);
    for len in [0, 1, 31, 32, 33, 64, 95, 96, 97, 1000, 1024, 4099] {
        let bytes = rng.bytes(len);
        let mut h = HighwayHasher::new(key);
        h.update(&bytes);
        assert_eq!(fairway64(key, &bytes), h.finalize64(), "64 at {len}");
        assert_eq!(fairway128(key, &bytes), h.finalize128(), "128 at {len}");
        assert_eq!(fairway256(key, &bytes), h.finalize256(), "256 at {len}");
        assert_eq!(fairway64(key, &bytes), scalar::hash64(key, &bytes), "{len}");
    }
}

#[test]
fn keyed_hash_matches_one_shot_and_leaves_hasher_untouched() {
    let key = Key(KEY64);
    let mut rng = common::Rng(11);
    for backend in backends() {
        let fresh = HighwayHasher::with_backend(key, backend);
        for len in [0, 1, 31, 32, 33, 64, 95, 96, 97, 1000, 1024, 4099] {
            let bytes = rng.bytes(len);
            let mut expected = HighwayHasher::with_backend(key, backend);
            expected.update(&bytes);
            assert_eq!(fresh.hash64(&bytes), expected.finalize64(), "64 at {len}");
            assert_eq!(
                fresh.hash128(&bytes),
                expected.finalize128(),
                "128 at {len}"
            );
            assert_eq!(
                fresh.hash256(&bytes),
                expected.finalize256(),
                "256 at {len}"
            );
        }
        // A fresh hasher stays fresh after `hash*`.
        assert_eq!(fresh.finalize64(), fairway64(key, b""));

        // With a partial packet buffered, `hash*` continues from it without
        // changing the hasher.
        let mut prefix = HighwayHasher::with_backend(key, backend);
        prefix.update(b"prefix");
        let before = prefix.finalize256();
        for len in [0, 1, 25, 26, 27, 58, 59, 1000] {
            let bytes = rng.bytes(len);
            let mut expected = HighwayHasher::with_backend(key, backend);
            expected.update(b"prefix");
            expected.update(&bytes);
            assert_eq!(prefix.hash64(&bytes), expected.finalize64(), "64 at {len}");
            assert_eq!(
                prefix.hash128(&bytes),
                expected.finalize128(),
                "128 at {len}"
            );
            assert_eq!(
                prefix.hash256(&bytes),
                expected.finalize256(),
                "256 at {len}"
            );
        }
        assert_eq!(prefix.finalize256(), before);
    }
}

/// `flat[i] = rand() & 0xFF` after `srand(129)`, as glibc generates them —
/// the exact bytes `VerifyCat` hashes upstream.
const CAT_FLAT: [u8; 105] = [
    252, 176, 15, 47, 29, 32, 6, 245, 89, 169, 208, 11, 53, 99, 89, 70, 220, 137, 78, 25, 18, 132,
    53, 164, 102, 252, 64, 114, 148, 76, 14, 145, 253, 29, 192, 26, 61, 198, 15, 151, 111, 224,
    162, 164, 67, 252, 235, 31, 133, 57, 56, 151, 189, 110, 59, 36, 106, 123, 150, 254, 199, 164,
    143, 196, 194, 80, 222, 255, 22, 238, 150, 134, 206, 57, 42, 17, 53, 21, 48, 186, 79, 104, 81,
    12, 214, 140, 48, 64, 7, 199, 63, 206, 107, 206, 147, 45, 30, 113, 45, 53, 95, 195, 187, 45,
    252,
];

/// `VerifyCat<Result>` / `TestHighwayHashCat`.
///
/// Upstream runs `TestHighwayHashCat(key, flat, i)` for every `i` in
/// `0..kMaxSize`, splitting into three fragments each `< i / 3` bytes. Those
/// splits for `i = kMaxSize - 1` are a superset of the splits for every
/// smaller `i` (same bytes, same key), so checking that one `i` covers exactly
/// the same cases without repeating them ~100 times.
fn verify_cat(width: Width) {
    let key = Key([
        0x1F1E1D1C1B1A1918,
        0x1716151413121110,
        0x0F0E0D0C0B0A0908,
        0x0706050403020100,
    ]);
    let size = CAT_FLAT.len() - 1;
    let max_fragment_size = size / 3;
    for backend in backends() {
        let results: Vec<Vec<u64>> = (0..=size)
            .map(|i| hash(backend, key, &CAT_FLAT[..i], width))
            .collect();
        for size1 in 0..max_fragment_size {
            for size2 in 0..max_fragment_size {
                for size3 in 0..max_fragment_size {
                    let mut cat = HighwayHasher::with_backend(key, backend);
                    let mut pos = 0;
                    for fragment in [size1, size2, size3] {
                        cat.update(&CAT_FLAT[pos..pos + fragment]);
                        pos += fragment;
                    }
                    assert_eq!(
                        finalize(&cat, width),
                        results[pos],
                        "Cat mismatch at size {pos} ({size1}+{size2}+{size3}) for target {} ({width:?})",
                        backend.name()
                    );
                }
            }
        }
    }
}

#[test]
fn verify_cat_64() {
    verify_cat(Width::W64);
}

#[test]
fn verify_cat_128() {
    verify_cat(Width::W128);
}

#[test]
fn verify_cat_256() {
    verify_cat(Width::W256);
}

/// Rust-specific: `finalize*` borrows, so finalizing several widths from the
/// same hasher and then continuing to stream must not disturb the state.
#[test]
fn finalize_does_not_consume_state() {
    let key = Key(KEY64);
    let input = golden_input();
    for backend in backends() {
        for width in WIDTHS {
            let mut h = HighwayHasher::with_backend(key, backend);
            h.update(&input[..20]);
            for other in WIDTHS {
                let _ = finalize(&h, other);
            }
            h.update(&input[20..64]);
            assert_eq!(
                finalize(&h, width),
                known_good(width, 64),
                "{backend:?} {width:?}"
            );
        }
    }
}

/// Rust-specific: `reset` re-keys and drops buffered bytes, keeping the backend.
#[test]
fn reset_restarts_with_new_key() {
    let input = golden_input();
    for backend in backends() {
        let mut h = HighwayHasher::with_backend(Key([9; 4]), backend);
        h.update(&input[..45]);
        h.reset(Key::from(KEY64));
        assert_eq!(h.backend(), backend);
        h.update(&input[..33]);
        assert_eq!(h.finalize64(), EXPECTED64[33], "{backend:?}");
    }
}

/// Rust-specific: unavailable backends fall back to portable instead of
/// executing unsupported instructions.
#[test]
fn unavailable_backend_falls_back_to_portable() {
    let available = backends();
    for backend in common::ALL_BACKENDS.iter().copied() {
        let h = HighwayHasher::with_backend(Key(KEY64), backend);
        let expected = if available.contains(&backend) {
            backend
        } else {
            Backend::Portable
        };
        assert_eq!(h.backend(), expected);
    }
    assert!(available.contains(&fairway_hash::best_backend()));
}

#[test]
fn backend_names() {
    let names: Vec<&str> = common::ALL_BACKENDS.iter().map(|b| b.name()).collect();
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    assert_eq!(names, ["Portable", "SSE4.1", "AVX2", "AVX512VL", "NEON"]);
    #[cfg(not(all(target_arch = "x86_64", feature = "avx512")))]
    assert_eq!(names, ["Portable", "SSE4.1", "AVX2", "NEON"]);
}
