# Changelog

All notable changes to fairway are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
