//! Throughput benchmark, zero dependencies:
//!
//! ```text
//! cargo run --release --example bench
//! ```
//!
//! Times the full streaming path (`HighwayHasher::update` + `finalize64`)
//! for every backend on a range of input sizes. `with_backend` silently
//! falls back to the portable backend on CPUs without the requested
//! feature, so the reported name is the one actually in use.

use fairway_hash::{Backend, HighwayHasher, Key};
use std::hint::black_box;
use std::time::{Duration, Instant};

const KEY: Key = Key([1, 2, 3, 4]);
/// ~50 ms of hashing per measurement, long enough to smooth jitter.
const TARGET: Duration = Duration::from_millis(50);
const SIZES: &[usize] = &[32, 64, 1024, 65536, 1 << 20];

fn throughput(backend: Backend, bytes: &[u8]) -> f64 {
    // Warm-up (feature detection, caches, page faults).
    let mut h = HighwayHasher::with_backend(KEY, backend);
    h.update(bytes);
    black_box(h.finalize64());

    let mut hashes = 0u64;
    let mut elapsed = Duration::ZERO;
    while elapsed < TARGET {
        let batch = if bytes.len() >= 1 << 16 { 8 } else { 256 };
        let start = Instant::now();
        for _ in 0..batch {
            let mut h = HighwayHasher::with_backend(KEY, backend);
            h.update(black_box(bytes));
            black_box(h.finalize64());
        }
        elapsed += start.elapsed();
        hashes += batch as u64;
    }
    hashes as f64 * bytes.len() as f64 / elapsed.as_secs_f64()
}

fn main() {
    let data: Vec<u8> = (0..SIZES.iter().copied().max().unwrap())
        .map(|i| i as u8)
        .collect();
    println!("fairway-hash throughput (streaming update + finalize64)");
    println!("{:>10} {:>10} {:>12}", "bytes", "backend", "MiB/s");
    for &backend in [Backend::Avx2, Backend::Sse41, Backend::Portable].iter() {
        for &size in SIZES {
            let mib_s = throughput(backend, &data[..size]) / (1024.0 * 1024.0);
            println!("{:>10} {:>10} {:>12.0}", size, backend.name(), mib_s);
        }
    }
}
