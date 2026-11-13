# Changelog

All notable changes to fairway are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Added

- `examples/bench.rs`, a zero-dependency throughput benchmark
  (`cargo run --release --example bench`).

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
