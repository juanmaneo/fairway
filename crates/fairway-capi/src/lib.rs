//! C ABI for [`fairway_hash`]: the functions declared by upstream
//! HighwayHash's `c_bindings.h`, under the same names, plus
//! `HighwayHash64_TargetNEON`. The matching header is
//! `include/fairway_capi.h`.
//!
//! Build the static or shared library with
//! `cargo build --release -p fairway-capi`.
//!
//! The results are the same as upstream's. One difference: upstream's
//! `HighwayHash64_Target*` run that instruction set unconditionally and crash
//! on a CPU without it; these fall back to the portable code, as
//! [`fairway_hash::HighwayHasher::with_backend`] does. Fairway has no VSX
//! backend, so `HighwayHash64_TargetVSX` is always the portable code.

use core::ffi::c_char;
use core::slice;

use fairway_hash::{sip_hash, sip_hash13, Backend, HighwayHasher, Key, SipHashKey};

/// Reads `N` key words from `key`, which need not be aligned.
///
/// # Safety
///
/// `key` must point to `N` readable `u64`s.
unsafe fn read_key<const N: usize>(key: *const u64) -> [u64; N] {
    // SAFETY: the caller guarantees `N` readable words; `read_unaligned`
    // needs no alignment (upstream requires none for SipHash keys).
    unsafe { key.cast::<[u64; N]>().read_unaligned() }
}

/// The input as a slice. `size == 0` accepts any pointer, null included.
///
/// # Safety
///
/// If `size > 0`, `bytes` must point to `size` readable bytes that stay
/// unchanged for `'a`.
unsafe fn input<'a>(bytes: *const c_char, size: u64) -> &'a [u8] {
    if size == 0 {
        return &[];
    }
    // SAFETY: the caller guarantees `size` readable bytes.
    unsafe { slice::from_raw_parts(bytes.cast::<u8>(), size as usize) }
}

/// # Safety
///
/// `key` must point to 4 readable `u64`s and `bytes` to `size` readable bytes
/// (`bytes` may be null if `size` is 0).
unsafe fn highway_hash64(
    backend: Backend,
    key: *const u64,
    bytes: *const c_char,
    size: u64,
) -> u64 {
    // SAFETY: forwarded from the caller.
    let (key, bytes) = unsafe { (read_key::<4>(key), input(bytes, size)) };
    HighwayHasher::with_backend(Key(key), backend).hash64(bytes)
}

/// 64-bit SipHash-2-4 (`SipHashC`).
///
/// # Safety
///
/// `key` must point to 2 readable `u64`s and `bytes` to `size` readable bytes
/// (`bytes` may be null if `size` is 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SipHashC(key: *const u64, bytes: *const c_char, size: u64) -> u64 {
    // SAFETY: forwarded from the caller.
    let (key, bytes) = unsafe { (read_key::<2>(key), input(bytes, size)) };
    sip_hash(SipHashKey(key), bytes)
}

/// 64-bit SipHash-1-3 (`SipHash13C`).
///
/// # Safety
///
/// As for [`SipHashC`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn SipHash13C(key: *const u64, bytes: *const c_char, size: u64) -> u64 {
    // SAFETY: forwarded from the caller.
    let (key, bytes) = unsafe { (read_key::<2>(key), input(bytes, size)) };
    sip_hash13(SipHashKey(key), bytes)
}

/// 64-bit HighwayHash on the best backend for the current CPU
/// (`HighwayHash64`).
///
/// # Safety
///
/// `key` must point to 4 readable `u64`s and `bytes` to `size` readable bytes
/// (`bytes` may be null if `size` is 0).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn HighwayHash64(key: *const u64, bytes: *const c_char, size: u64) -> u64 {
    // SAFETY: forwarded from the caller.
    unsafe { highway_hash64(fairway_hash::best_backend(), key, bytes, size) }
}

/// 64-bit HighwayHash on the portable code (`HighwayHash64_TargetPortable`).
///
/// # Safety
///
/// As for [`HighwayHash64`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn HighwayHash64_TargetPortable(
    key: *const u64,
    bytes: *const c_char,
    size: u64,
) -> u64 {
    // SAFETY: forwarded from the caller.
    unsafe { highway_hash64(Backend::Portable, key, bytes, size) }
}

/// 64-bit HighwayHash on SSE4.1, or the portable code where that is not
/// available (`HighwayHash64_TargetSSE41`).
///
/// # Safety
///
/// As for [`HighwayHash64`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn HighwayHash64_TargetSSE41(
    key: *const u64,
    bytes: *const c_char,
    size: u64,
) -> u64 {
    // SAFETY: forwarded from the caller.
    unsafe { highway_hash64(Backend::Sse41, key, bytes, size) }
}

/// 64-bit HighwayHash on AVX2, or the portable code where that is not
/// available (`HighwayHash64_TargetAVX2`).
///
/// # Safety
///
/// As for [`HighwayHash64`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn HighwayHash64_TargetAVX2(
    key: *const u64,
    bytes: *const c_char,
    size: u64,
) -> u64 {
    // SAFETY: forwarded from the caller.
    unsafe { highway_hash64(Backend::Avx2, key, bytes, size) }
}

/// 64-bit HighwayHash on NEON, or the portable code where that is not
/// available (`HighwayHash64_TargetNEON`; not in upstream's header).
///
/// # Safety
///
/// As for [`HighwayHash64`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn HighwayHash64_TargetNEON(
    key: *const u64,
    bytes: *const c_char,
    size: u64,
) -> u64 {
    // SAFETY: forwarded from the caller.
    unsafe { highway_hash64(Backend::Neon, key, bytes, size) }
}

/// Always the portable code: fairway has no VSX backend
/// (`HighwayHash64_TargetVSX`, kept so upstream's header links).
///
/// # Safety
///
/// As for [`HighwayHash64`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn HighwayHash64_TargetVSX(
    key: *const u64,
    bytes: *const c_char,
    size: u64,
) -> u64 {
    // SAFETY: forwarded from the caller.
    unsafe { highway_hash64(Backend::Portable, key, bytes, size) }
}
