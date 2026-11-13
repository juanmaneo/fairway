//! Throughput benchmark, zero dependencies:
//!
//! ```text
//! cargo run --release --example bench
//! ```
//!
//! Two parts:
//!
//! 1. Single input: the full streaming path (`HighwayHasher::update` +
//!    `finalize64`) for every backend on a range of input sizes.
//!    `with_backend` silently falls back to the portable backend on CPUs
//!    without the requested feature, so the reported name is the one actually
//!    in use.
//! 2. Many independent inputs: a plain loop over `fairway64` versus
//!    `fairway64_many` (interleaved) and `fairway64_many_par` (interleaved and
//!    multi-threaded).

use fairway_hash::{fairway64, fairway64_many, fairway64_many_par, Backend, HighwayHasher, Key};
use std::hint::black_box;
use std::time::{Duration, Instant};

const KEY: Key = Key([1, 2, 3, 4]);
/// ~50 ms of hashing per measurement, long enough to smooth jitter.
const TARGET: Duration = Duration::from_millis(50);
const SIZES: &[usize] = &[32, 64, 1024, 65536, 1 << 20];
/// (number of inputs, bytes per input) for the batch benchmark.
const BATCHES: &[(usize, usize)] = &[
    (4096, 32),
    (4096, 64),
    (1024, 1024),
    (256, 16 * 1024),
    (64, 256 * 1024),
    (256, 256 * 1024),
];

/// Runs `f` repeatedly for about `TARGET` and returns MiB/s, given the number
/// of bytes each call hashes.
fn measure(bytes_per_call: usize, mut f: impl FnMut()) -> f64 {
    f(); // Warm-up (feature detection, caches, page faults, thread start).
    let mut calls = 0u64;
    let start = Instant::now();
    while start.elapsed() < TARGET {
        f();
        calls += 1;
    }
    calls as f64 * bytes_per_call as f64 / start.elapsed().as_secs_f64() / (1024.0 * 1024.0)
}

fn single_input(backend: Backend, bytes: &[u8]) -> f64 {
    // Many hashes per timed call for small inputs, so the clock is cheap.
    let batch = (1 << 16) / bytes.len().max(1) + 1;
    measure(bytes.len() * batch, || {
        for _ in 0..batch {
            let mut h = HighwayHasher::with_backend(KEY, backend);
            h.update(black_box(bytes));
            black_box(h.finalize64());
        }
    })
}

fn main() {
    let data: Vec<u8> = (0..SIZES.iter().copied().max().unwrap())
        .map(|i| i as u8)
        .collect();
    println!("single input (streaming update + finalize64)");
    println!("{:>10} {:>10} {:>12}", "bytes", "backend", "MiB/s");
    let backends = [
        Backend::Avx2,
        Backend::Sse41,
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        Backend::Avx512,
        Backend::Portable,
    ];
    for &backend in backends.iter() {
        for &size in SIZES {
            let mib_s = single_input(backend, &data[..size]);
            println!("{:>10} {:>10} {:>12.0}", size, backend.name(), mib_s);
        }
    }

    println!();
    println!(
        "many inputs (best backend: {})",
        fairway_hash::best_backend().name()
    );
    println!(
        "{:>8} {:>9} {:>12} {:>12} {:>12}",
        "inputs", "bytes", "loop MiB/s", "many MiB/s", "many_par"
    );
    for &(count, size) in BATCHES {
        let inputs: Vec<Vec<u8>> = (0..count).map(|i| data[i..i + size].to_vec()).collect();
        let mut out = vec![0u64; count];
        let total = count * size;

        let looped = measure(total, || {
            for (input, slot) in black_box(&inputs).iter().zip(&mut out) {
                *slot = fairway64(KEY, input);
            }
            black_box(&mut out);
        });
        let many = measure(total, || {
            fairway64_many(KEY, black_box(&inputs), &mut out);
            black_box(&mut out);
        });
        let par = measure(total, || {
            fairway64_many_par(KEY, black_box(&inputs), &mut out);
            black_box(&mut out);
        });
        println!(
            "{:>8} {:>9} {:>12.0} {:>12.0} {:>12.0}",
            count, size, looped, many, par
        );
    }
}
