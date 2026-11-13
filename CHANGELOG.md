# Changelog

All notable changes to fairway are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Small inputs on the AVX2 backend are fast without link-time optimization.
  The backend's `new`, `update_remainder` and `finalize*` functions now carry
  `#[target_feature(enable = "avx2")]` themselves, so the compiler inlines them
  into the entry points with the default release profile. Measured on a Zen 3
  laptop with `lto = false`: a 32-byte hash went from about 32 MiB/s to about
  450 MiB/s, a 1 KiB hash from about 0.9 GiB/s to about 4.2 GiB/s, and large
  inputs are unchanged (about 5 GiB/s). With LTO enabled nothing changes. The
  0.1.1 "Known limitations" note no longer applies. Hash output is unchanged.
- `fairway*_many` interleave six inputs instead of four. Measured on Zen 3
  (AVX2) against four and eight lanes, six was about 5-8% faster on inputs
  of 1 KiB and up, and eight was slower than four.

### Added

- `HighwayHasher::hash64`, `hash128` and `hash256`: hash an input starting
  from the hasher's current state without modifying it. On a fresh hasher
  this equals the one-shot functions with the key schedule and CPU detection
  done once, which is about 1.5x faster for inputs under 1 KiB when many
  inputs share one key (22 million 32-byte hashes per second on one Zen 3
  core, against 15 million for `fairway64`).
- Optional AVX-512VL backend on x86_64 behind the new off-by-default `avx512`
  cargo feature: the AVX2 algorithm re-encoded with EVEX instructions on
  256-bit registers (`_mm256_rolv_epi32`, `_mm256_ternarylogic_epi32`,
  `_mm256_maskz_loadu_epi8`). Selected at runtime when
  avx512f+avx512vl+avx512bw are detected, forcible with
  `HighwayHasher::with_backend(_, Backend::Avx512)` (falls back to portable
  where unsupported). Hash output is unchanged: the golden vectors pass on
  AVX-512 hardware, and the fuzzer cross-checks it against the portable
  backend. Enabling the feature requires Rust 1.89+, where the AVX-512
  intrinsics and target features were stabilized; the crate's MSRV stays
  1.88 without it. Measured on an Emerald Rapids Xeon 6985P (one core, LTO
  off, 64 MiB per run, hyperfine, 15 runs): not a win for the common one-shot
  path there — `fairway64` on 32-byte inputs is about 25% slower than AVX2
  (64 vs 81 ms per 64 MiB), about 10% slower at 1 KiB, equal at 1 MiB — and
  `fairway64_many` is much slower on 32-byte batches (46 vs 71 ms) but about
  18% faster on 1 KiB batches (43 vs 36 ms). It therefore stays opt-in; AVX2
  remains the default choice on this hardware.

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
