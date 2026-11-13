# Changelog

All notable changes to fairway are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-11-13

Performance release. Hash output is unchanged and the public API only grows.

### Changed

- Much faster AVX2 and SSE4.1 backends. The `core::arch` intrinsics were
  being called out of line because the calling code did not enable the
  matching CPU feature, which made AVX2 slower than the portable code. The
  hot path now goes through `#[target_feature]` entry points that let the
  compiler inline the whole update loop. On a Zen 3 laptop, AVX2 throughput
  on large inputs went from about 150 MiB/s to about 6.9 GiB/s, and a 32-byte
  hash from about 26 MiB/s to about 1 GiB/s. Hash output is unchanged.
- `HighwayHasher::update` dispatches on the backend once per call instead of
  once per 32-byte packet, and no longer copies the completed packet through
  a temporary buffer.
- `fairway64` / `fairway128` / `fairway256` feed whole packets straight from
  the input and the tail straight into finalization, instead of going through
  the incremental hasher and its 32-byte buffer.

### Added

- Batch hashing of many independent inputs: `fairway64_many`,
  `fairway128_many` and `fairway256_many`. A single hash is a serial chain,
  so the CPU idles between steps; four inputs are advanced together to fill
  those gaps. Output is identical to calling the one-shot functions in a
  loop, and any byte-like input type works (`&[u8]`, `Vec<u8>`, `String`).
  On a Zen 3 laptop this is about 1.4x faster than a loop for inputs of
  several KiB and larger (9.5 GiB/s against 6.9 GiB/s), 1.2-1.5x from 256
  bytes to a few KiB, and about 2x for inputs of 64 bytes or less (one key
  schedule for the whole batch).
- `fairway64_many_par`, `fairway128_many_par` and `fairway256_many_par`
  (`std` feature): the same, spread over scoped threads when the batch is
  large enough to pay for them (16-24 GiB/s on a 16-thread laptop for batches of
  16-128 MiB, against 6.8 GiB/s for a loop). The crate still has no dependencies.
- `examples/bench.rs`, a zero-dependency throughput benchmark
  (`cargo run --release --example bench`).

### Known limitations

- The speedups above assume the compiler inlines the AVX2 / SSE4.1 backends
  into the entry points. That always happens for the update loop (large
  inputs reach about 6.5 GiB/s with the default release profile), but for
  *small* inputs on AVX2 the finalization code is only inlined with link-time
  optimization (`lto = true`) or `-C target-cpu=native` / `-C
  target-feature=+avx2`. Without either, a 32-byte AVX2 hash still runs at
  about 30 MiB/s, as in 0.1.0. Applications that hash many small inputs
  with the AVX2 backend should enable one of them.

## [0.1.0] - 2026-11-12

Initial release: a zero-dependency Rust port of Google's HighwayHash.

### Added

- `fairway-hash`: 64/128/256-bit HighwayHash and SipHash-2-4 / SipHash-1-3,
  verified against the upstream C++ golden test vectors (all input sizes
  0..=64, the SipHash reference vectors, and the streaming "cat"
  equivalence test).
- Backends: AVX2 and SSE4.1 on x86_64 (runtime-detected), NEON on aarch64,
  and a portable scalar fallback everywhere else, including big-endian and
  `no_std`.
