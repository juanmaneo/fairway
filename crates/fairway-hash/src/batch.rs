//! Hashing many independent inputs at once.
//!
//! Hashing one input is a serial dependency chain: every 32-byte packet
//! updates a state that the next packet needs, so a CPU core spends most of
//! its time waiting on its own previous result. Several *independent* inputs
//! have no such dependency, so [`LANES`] of them are advanced in lockstep and
//! their instructions overlap. The results are identical to hashing each
//! input on its own.

use crate::{best_backend, Backend, Inner, Key};

#[cfg(all(
    any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_endian = "little")
    ),
    not(feature = "portable-only")
))]
use crate::arch;
use crate::portable;

/// Number of inputs advanced together. Six independent chains hide the
/// multiply latency; measured against four and eight on Zen 3 (AVX2), six
/// was fastest and eight was slower than four (register spills).
const LANES: usize = 6;

/// Fewer common packets than this and interleaving costs more than it saves
/// (setting up and splitting a group has a fixed price), so the inputs are
/// hashed one at a time instead.
const MIN_LOCKSTEP_PACKETS: usize = 16;

/// Below this much data per thread, starting a thread costs more than it
/// saves.
#[cfg(feature = "std")]
const MIN_BYTES_PER_THREAD: usize = 4 * 1024 * 1024;

/// [`LANES`] hash states of one backend, advanced together.
enum Group {
    Portable([portable::State; LANES]),
    #[cfg(all(
        target_arch = "x86_64",
        feature = "avx512",
        not(feature = "portable-only")
    ))]
    Avx512([arch::x86::avx512::State; LANES]),
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
    Avx2([arch::x86::avx2::State; LANES]),
    #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
    Sse41([arch::x86::sse41::State; LANES]),
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        not(feature = "portable-only")
    ))]
    Neon([arch::aarch64::neon::State; LANES]),
}

impl Group {
    /// [`LANES`] copies of `fresh`, which must be a state that has not
    /// absorbed any input yet.
    fn new(fresh: &Inner) -> Self {
        match fresh {
            Inner::Portable(s) => Group::Portable(core::array::from_fn(|_| s.clone())),
            #[cfg(all(
                target_arch = "x86_64",
                feature = "avx512",
                not(feature = "portable-only")
            ))]
            Inner::Avx512(s) => Group::Avx512([*s; LANES]),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Avx2(s) => Group::Avx2([*s; LANES]),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Inner::Sse41(s) => Group::Sse41([*s; LANES]),
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                not(feature = "portable-only")
            ))]
            Inner::Neon(s) => Group::Neon([*s; LANES]),
        }
    }

    /// Feeds packet `p` of every stream to its state, for every `p`, one
    /// packet of each stream before moving on. The streams must have the
    /// same length.
    fn update_lockstep(&mut self, streams: &[&[[u8; 32]]; LANES]) {
        match self {
            Group::Portable(states) => lockstep(states, streams, |s, p| s.update_packet(p)),
            #[cfg(all(
                target_arch = "x86_64",
                feature = "avx512",
                not(feature = "portable-only")
            ))]
            Group::Avx512(states) => {
                // SAFETY: a `Group::Avx512` exists only when AVX-512 was detected.
                unsafe { arch::x86::entry::avx512::update_lockstep(states, streams) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Group::Avx2(states) => {
                // SAFETY: a `Group::Avx2` exists only when AVX2 was detected.
                unsafe { arch::x86::entry::avx2::update_lockstep(states, streams) }
            }
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Group::Sse41(states) => {
                // SAFETY: a `Group::Sse41` exists only when SSE4.1 was detected.
                unsafe { arch::x86::entry::sse41::update_lockstep(states, streams) }
            }
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                not(feature = "portable-only")
            ))]
            Group::Neon(states) => lockstep(states, streams, |s, p| s.update_packet(p)),
        }
    }

    /// A copy of the state of `lane` as an independent hasher state.
    fn lane(&self, lane: usize) -> Inner {
        match self {
            Group::Portable(states) => Inner::Portable(states[lane].clone()),
            #[cfg(all(
                target_arch = "x86_64",
                feature = "avx512",
                not(feature = "portable-only")
            ))]
            Group::Avx512(states) => Inner::Avx512(states[lane]),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Group::Avx2(states) => Inner::Avx2(states[lane]),
            #[cfg(all(target_arch = "x86_64", not(feature = "portable-only")))]
            Group::Sse41(states) => Inner::Sse41(states[lane]),
            #[cfg(all(
                target_arch = "aarch64",
                target_endian = "little",
                not(feature = "portable-only")
            ))]
            Group::Neon(states) => Inner::Neon(states[lane]),
        }
    }
}

/// Advances `states[i]` over `streams[i]`, interleaving the streams packet by
/// packet. Only the first `min(len)` packets of each stream are consumed.
fn lockstep<S>(
    states: &mut [S; LANES],
    streams: &[&[[u8; 32]]; LANES],
    mut update: impl FnMut(&mut S, &[u8; 32]),
) {
    let packets = streams.iter().map(|s| s.len()).min().unwrap_or(0);
    for p in 0..packets {
        for (state, stream) in states.iter_mut().zip(streams) {
            update(state, &stream[p]);
        }
    }
}

/// Hashes one input from a fresh state.
fn hash_one<T>(fresh: &Inner, input: &[u8], finish: fn(&Inner, &[u8]) -> T) -> T {
    let (packets, remainder) = input.as_chunks::<32>();
    let mut inner = fresh.clone();
    inner.update_packets(packets);
    finish(&inner, remainder)
}

/// Hashes `ins` together: the packets they all have in common are processed
/// in lockstep, then each input finishes on its own (so inputs of different
/// lengths are fine, merely less parallel).
fn hash_group<I: AsRef<[u8]>, T: Copy>(
    fresh: &Inner,
    ins: &[I; LANES],
    outs: &mut [T; LANES],
    finish: fn(&Inner, &[u8]) -> T,
) {
    let split: [(&[[u8; 32]], &[u8]); LANES] =
        core::array::from_fn(|i| ins[i].as_ref().as_chunks::<32>());
    let common = split
        .iter()
        .map(|(packets, _)| packets.len())
        .min()
        .unwrap_or(0);
    if common < MIN_LOCKSTEP_PACKETS {
        for (input, out) in ins.iter().zip(outs) {
            *out = hash_one(fresh, input.as_ref(), finish);
        }
        return;
    }
    let mut group = Group::new(fresh);
    group.update_lockstep(&core::array::from_fn(|i| &split[i].0[..common]));
    for (lane, ((packets, remainder), out)) in split.iter().zip(outs).enumerate() {
        let mut inner = group.lane(lane);
        inner.update_packets(&packets[common..]);
        *out = finish(&inner, remainder);
    }
}

/// Sequential batch hashing: groups of [`LANES`] inputs interleaved, then the
/// leftover inputs one by one.
fn hash_many<I: AsRef<[u8]>, T: Copy>(
    backend: Backend,
    key: Key,
    inputs: &[I],
    out: &mut [T],
    finish: fn(&Inner, &[u8]) -> T,
) {
    assert_eq!(
        inputs.len(),
        out.len(),
        "`inputs` and `out` must have the same length"
    );
    // The key schedule and CPU detection happen once, not once per group.
    let fresh = Inner::new(key, backend);
    let (in_groups, in_rest) = inputs.as_chunks::<LANES>();
    let (out_groups, out_rest) = out.as_chunks_mut::<LANES>();
    for (ins, outs) in in_groups.iter().zip(out_groups) {
        hash_group(&fresh, ins, outs, finish);
    }
    for (input, slot) in in_rest.iter().zip(out_rest) {
        *slot = hash_one(&fresh, input.as_ref(), finish);
    }
}

/// Like [`hash_many`], but spreads the inputs over `threads` scoped threads,
/// each taking a run of consecutive inputs holding about the same number of
/// bytes. With `threads <= 1` nothing is spawned.
#[cfg(feature = "std")]
fn hash_many_threaded<I: AsRef<[u8]> + Sync, T: Copy + Send>(
    backend: Backend,
    key: Key,
    inputs: &[I],
    out: &mut [T],
    finish: fn(&Inner, &[u8]) -> T,
    threads: usize,
) {
    if threads <= 1 {
        return hash_many(backend, key, inputs, out, finish);
    }
    assert_eq!(
        inputs.len(),
        out.len(),
        "`inputs` and `out` must have the same length"
    );
    let total: usize = inputs.iter().map(|i| i.as_ref().len()).sum();
    let per_thread = total.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        let (mut inputs, mut out) = (inputs, out);
        while !inputs.is_empty() {
            let (mut n, mut bytes) = (0, 0);
            while n < inputs.len() && bytes < per_thread {
                bytes += inputs[n].as_ref().len();
                n += 1;
            }
            let (chunk_in, rest_in) = inputs.split_at(n);
            let (chunk_out, rest_out) = core::mem::take(&mut out).split_at_mut(n);
            if rest_in.is_empty() {
                // The calling thread would only wait, so it takes the last run.
                hash_many(backend, key, chunk_in, chunk_out, finish);
            } else {
                scope.spawn(move || hash_many(backend, key, chunk_in, chunk_out, finish));
            }
            (inputs, out) = (rest_in, rest_out);
        }
    });
}

/// How many threads are worth using for `total_bytes` of input spread over
/// `inputs` inputs, given `available` hardware threads. At least 1.
#[cfg(feature = "std")]
fn thread_count(total_bytes: usize, inputs: usize, available: usize) -> usize {
    (total_bytes / MIN_BYTES_PER_THREAD)
        .min(inputs)
        .min(available)
        .max(1)
}

#[cfg(feature = "std")]
fn hash_many_parallel<I: AsRef<[u8]> + Sync, T: Copy + Send>(
    key: Key,
    inputs: &[I],
    out: &mut [T],
    finish: fn(&Inner, &[u8]) -> T,
) {
    let total: usize = inputs.iter().map(|i| i.as_ref().len()).sum();
    // Asking the OS how many threads there are is a system call that costs
    // more than hashing a small batch, so only do it when the batch is big
    // enough for more than one thread.
    let threads = if thread_count(total, inputs.len(), usize::MAX) > 1 {
        let available = std::thread::available_parallelism().map_or(1, core::num::NonZero::get);
        thread_count(total, inputs.len(), available)
    } else {
        1
    };
    hash_many_threaded(best_backend(), key, inputs, out, finish, threads);
}

/// Hashes every input with the 64-bit HighwayHash, writing `out[i] =
/// fairway64(key, inputs[i])`.
///
/// Faster than calling [`fairway64`](crate::fairway64) in a loop: independent
/// inputs are processed several at a time, which keeps the CPU busy while it
/// would otherwise wait on each hash's serial dependency chain. The gain is
/// largest when the inputs have similar lengths, and the results are always
/// identical to the one-at-a-time functions.
///
/// `I` can be anything byte-like: `&[u8]`, `Vec<u8>`, `String`, ...
///
/// # Panics
///
/// Panics if `inputs` and `out` have different lengths.
///
/// # Examples
///
/// ```
/// use fairway_hash::{fairway64, fairway64_many, Key};
///
/// let key = Key([1, 2, 3, 4]);
/// let inputs = [&b"alpha"[..], b"beta", b"gamma"];
/// let mut out = [0u64; 3];
/// fairway64_many(key, &inputs, &mut out);
/// assert_eq!(out[1], fairway64(key, b"beta"));
/// ```
pub fn fairway64_many<I: AsRef<[u8]>>(key: Key, inputs: &[I], out: &mut [u64]) {
    hash_many(best_backend(), key, inputs, out, Inner::finalize64);
}

/// 128-bit variant of [`fairway64_many`]: `out[i] = fairway128(key,
/// inputs[i])`.
///
/// # Panics
///
/// Panics if `inputs` and `out` have different lengths.
pub fn fairway128_many<I: AsRef<[u8]>>(key: Key, inputs: &[I], out: &mut [[u64; 2]]) {
    hash_many(best_backend(), key, inputs, out, Inner::finalize128);
}

/// 256-bit variant of [`fairway64_many`]: `out[i] = fairway256(key,
/// inputs[i])`.
///
/// # Panics
///
/// Panics if `inputs` and `out` have different lengths.
pub fn fairway256_many<I: AsRef<[u8]>>(key: Key, inputs: &[I], out: &mut [[u64; 4]]) {
    hash_many(best_backend(), key, inputs, out, Inner::finalize256);
}

/// Multi-threaded [`fairway64_many`] (needs the `std` feature).
///
/// Large batches are split over scoped threads; small ones run on the calling
/// thread, because starting threads only pays off from a few MiB of
/// data per thread. Hashing a *single* input cannot be parallelized, so this
/// helps only when there are many inputs. The results are identical to
/// [`fairway64_many`].
///
/// # Panics
///
/// Panics if `inputs` and `out` have different lengths.
#[cfg(feature = "std")]
pub fn fairway64_many_par<I: AsRef<[u8]> + Sync>(key: Key, inputs: &[I], out: &mut [u64]) {
    hash_many_parallel(key, inputs, out, Inner::finalize64);
}

/// Multi-threaded [`fairway128_many`]; see [`fairway64_many_par`].
///
/// # Panics
///
/// Panics if `inputs` and `out` have different lengths.
#[cfg(feature = "std")]
pub fn fairway128_many_par<I: AsRef<[u8]> + Sync>(key: Key, inputs: &[I], out: &mut [[u64; 2]]) {
    hash_many_parallel(key, inputs, out, Inner::finalize128);
}

/// Multi-threaded [`fairway256_many`]; see [`fairway64_many_par`].
///
/// # Panics
///
/// Panics if `inputs` and `out` have different lengths.
#[cfg(feature = "std")]
pub fn fairway256_many_par<I: AsRef<[u8]> + Sync>(key: Key, inputs: &[I], out: &mut [[u64; 4]]) {
    hash_many_parallel(key, inputs, out, Inner::finalize256);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar;

    const KEY: Key = Key([
        0x0706050403020100,
        0x0f0e0d0c0b0a0908,
        0x1716151413121110,
        0x1f1e1d1c1b1a1918,
    ]);

    /// Every backend this build knows about; unsupported ones fall back to
    /// portable, so this is valid on any CPU.
    fn backends() -> Vec<Backend> {
        [
            Backend::Avx2,
            Backend::Sse41,
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            Backend::Avx512,
            Backend::Neon,
            Backend::Portable,
        ]
        .into_iter()
        .collect()
    }

    fn data(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
            .collect()
    }

    /// Batches that exercise: fewer inputs than a group, exactly one group,
    /// a group plus leftovers, equal and very different lengths, and empty
    /// inputs.
    fn batches() -> Vec<Vec<Vec<u8>>> {
        let lens: &[&[usize]] = &[
            &[],
            &[5],
            &[0, 0, 0],
            &[32, 32, 32, 32],
            &[31, 32, 33, 64],
            &[64; 5],
            &[100, 3, 64, 0, 7, 500, 32, 33],
            &[1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
            &[4096, 1, 1, 1, 65, 65, 65, 65, 2000, 2000, 2000],
            // Groups with a long common prefix but unequal lengths.
            &[2000, 1500, 3000, 600, 1000, 900, 2500, 1100, 777],
        ];
        lens.iter()
            .map(|lens| {
                lens.iter()
                    .enumerate()
                    .map(|(i, &len)| data(len, i as u8))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn many_matches_single_hashes_on_every_backend() {
        for backend in backends() {
            for batch in batches() {
                let mut out64 = vec![0u64; batch.len()];
                hash_many(backend, KEY, &batch, &mut out64, Inner::finalize64);
                let mut out128 = vec![[0u64; 2]; batch.len()];
                hash_many(backend, KEY, &batch, &mut out128, Inner::finalize128);
                let mut out256 = vec![[0u64; 4]; batch.len()];
                hash_many(backend, KEY, &batch, &mut out256, Inner::finalize256);
                for (i, input) in batch.iter().enumerate() {
                    let lens: Vec<usize> = batch.iter().map(Vec::len).collect();
                    let ctx = format!("{} input {i} of {lens:?}", backend.name());
                    assert_eq!(out64[i], scalar::hash64(KEY, input), "64: {ctx}");
                    assert_eq!(out128[i], scalar::hash128(KEY, input), "128: {ctx}");
                    assert_eq!(out256[i], scalar::hash256(KEY, input), "256: {ctx}");
                }
            }
        }
    }

    #[test]
    fn public_functions_accept_byte_like_inputs() {
        let owned: Vec<Vec<u8>> = (0..7).map(|i| data(i * 37, i as u8)).collect();
        let slices: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let strings: Vec<String> = ["", "a", "hello", "world!", "fairway"]
            .iter()
            .map(|s| s.to_string())
            .collect();

        let mut a = vec![0u64; owned.len()];
        let mut b = vec![0u64; owned.len()];
        fairway64_many(KEY, &owned, &mut a);
        fairway64_many(KEY, &slices, &mut b);
        let expected: Vec<u64> = owned.iter().map(|x| crate::fairway64(KEY, x)).collect();
        assert_eq!(a, expected);
        assert_eq!(b, expected);

        let mut c = vec![[0u64; 2]; strings.len()];
        fairway128_many(KEY, &strings, &mut c);
        let expected: Vec<[u64; 2]> = strings
            .iter()
            .map(|x| crate::fairway128(KEY, x.as_bytes()))
            .collect();
        assert_eq!(c, expected);

        let mut d = vec![[0u64; 4]; strings.len()];
        fairway256_many(KEY, &strings, &mut d);
        let expected: Vec<[u64; 4]> = strings
            .iter()
            .map(|x| crate::fairway256(KEY, x.as_bytes()))
            .collect();
        assert_eq!(d, expected);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn mismatched_output_length_panics() {
        let mut out = [0u64; 2];
        fairway64_many(KEY, &[b"a", b"b", b"c"], &mut out);
    }

    #[cfg(feature = "std")]
    mod threaded {
        use super::*;

        #[test]
        fn threaded_matches_sequential_for_any_thread_count() {
            for backend in backends() {
                for batch in batches() {
                    let mut expected = vec![0u64; batch.len()];
                    hash_many(backend, KEY, &batch, &mut expected, Inner::finalize64);
                    for threads in [0, 1, 2, 3, 5, 64] {
                        let mut out = vec![0u64; batch.len()];
                        hash_many_threaded(
                            backend,
                            KEY,
                            &batch,
                            &mut out,
                            Inner::finalize64,
                            threads,
                        );
                        assert_eq!(out, expected, "{} threads {threads}", backend.name());
                    }
                }
            }
        }

        #[test]
        fn threaded_handles_all_empty_inputs() {
            let batch = vec![Vec::<u8>::new(); 9];
            let mut out = vec![1u64; 9];
            hash_many_threaded(
                Backend::Portable,
                KEY,
                &batch,
                &mut out,
                Inner::finalize64,
                4,
            );
            assert_eq!(out, vec![scalar::hash64(KEY, b""); 9]);
        }

        #[test]
        #[should_panic(expected = "same length")]
        fn threaded_mismatched_output_length_panics() {
            let mut out = [0u64; 1];
            hash_many_threaded(
                Backend::Portable,
                KEY,
                &[b"a", b"b"],
                &mut out,
                Inner::finalize64,
                2,
            );
        }

        #[test]
        fn thread_count_scales_with_data_and_is_bounded() {
            // Too little data: stay on the calling thread.
            assert_eq!(thread_count(0, 0, 16), 1);
            assert_eq!(thread_count(MIN_BYTES_PER_THREAD - 1, 100, 16), 1);
            // One thread per MIN_BYTES_PER_THREAD ...
            assert_eq!(thread_count(3 * MIN_BYTES_PER_THREAD, 100, 16), 3);
            // ... but never more than the inputs or the hardware offer.
            assert_eq!(thread_count(64 * MIN_BYTES_PER_THREAD, 2, 16), 2);
            assert_eq!(thread_count(64 * MIN_BYTES_PER_THREAD, 100, 4), 4);
        }

        #[test]
        fn public_parallel_functions_match_sequential() {
            // More than MIN_BYTES_PER_THREAD of data, so machines with more
            // than one hardware thread really spawn threads.
            let batch: Vec<Vec<u8>> = (0..12).map(|i| data(400 * 1024 + i, i as u8)).collect();
            let total: usize = batch.iter().map(Vec::len).sum();
            assert!(total > MIN_BYTES_PER_THREAD);

            let mut seq = vec![0u64; batch.len()];
            let mut par = vec![0u64; batch.len()];
            fairway64_many(KEY, &batch, &mut seq);
            fairway64_many_par(KEY, &batch, &mut par);
            assert_eq!(par, seq);

            let mut seq = vec![[0u64; 2]; batch.len()];
            let mut par = vec![[0u64; 2]; batch.len()];
            fairway128_many(KEY, &batch, &mut seq);
            fairway128_many_par(KEY, &batch, &mut par);
            assert_eq!(par, seq);

            let mut seq = vec![[0u64; 4]; batch.len()];
            let mut par = vec![[0u64; 4]; batch.len()];
            fairway256_many(KEY, &batch, &mut seq);
            fairway256_many_par(KEY, &batch, &mut par);
            assert_eq!(par, seq);
        }
    }
}
