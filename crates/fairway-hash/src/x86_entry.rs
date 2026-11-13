//! Feature-enabled entry points into the x86_64 SIMD backends.
//!
//! The `core::arch` intrinsics are `#[target_feature]` functions, and the
//! compiler only inlines them into callers that enable the same feature.
//! Without this, every `_mm256_*` call in the backends is compiled as an
//! out-of-line call (with the vectors round-tripping through memory), which
//! makes AVX2 several times *slower* than the portable code. Each function
//! below enables the feature itself, so the whole operation (and any loop
//! around it) compiles into straight-line SIMD code.
//!
//! Callers must only use a module after detecting its CPU feature.

use crate::Key;

macro_rules! entry_points {
    ($module:ident, $feature:literal, $state:ty) => {
        pub(crate) mod $module {
            use super::Key;

            type State = $state;

            /// # Safety
            /// CPU must support the feature this module is named after.
            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn new(key: Key) -> State {
                // SAFETY: the caller guarantees the CPU feature.
                unsafe { State::new(key) }
            }

            /// # Safety
            /// CPU must support the feature this module is named after.
            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn update_packets(state: &mut State, packets: &[[u8; 32]]) {
                for packet in packets {
                    // SAFETY: the caller guarantees the CPU feature.
                    unsafe { state.update_packet(packet) };
                }
            }

            /// Finalizes a copy of `state` after absorbing `remainder`
            /// (0..=31 bytes).
            ///
            /// # Safety
            /// CPU must support the feature this module is named after.
            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn finalize64(state: &State, remainder: &[u8]) -> u64 {
                let mut state = *state;
                // SAFETY: the caller guarantees the CPU feature, and
                // `remainder` has fewer than 32 bytes.
                unsafe {
                    if !remainder.is_empty() {
                        state.update_remainder(remainder);
                    }
                    state.finalize64()
                }
            }

            /// See [`finalize64`].
            ///
            /// # Safety
            /// CPU must support the feature this module is named after.
            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn finalize128(state: &State, remainder: &[u8]) -> [u64; 2] {
                let mut state = *state;
                // SAFETY: the caller guarantees the CPU feature, and
                // `remainder` has fewer than 32 bytes.
                unsafe {
                    if !remainder.is_empty() {
                        state.update_remainder(remainder);
                    }
                    state.finalize128()
                }
            }

            /// See [`finalize64`].
            ///
            /// # Safety
            /// CPU must support the feature this module is named after.
            #[target_feature(enable = $feature)]
            pub(crate) unsafe fn finalize256(state: &State, remainder: &[u8]) -> [u64; 4] {
                let mut state = *state;
                // SAFETY: the caller guarantees the CPU feature, and
                // `remainder` has fewer than 32 bytes.
                unsafe {
                    if !remainder.is_empty() {
                        state.update_remainder(remainder);
                    }
                    state.finalize256()
                }
            }
        }
    };
}

entry_points!(avx2, "avx2", crate::arch::x86::avx2::State);
entry_points!(sse41, "sse4.1", crate::arch::x86::sse41::State);
