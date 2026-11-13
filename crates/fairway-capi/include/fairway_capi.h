/* C interface to fairway-hash: upstream HighwayHash's c_bindings.h. */

#ifndef FAIRWAY_CAPI_H_
#define FAIRWAY_CAPI_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SipHash-2-4 and SipHash-1-3. `key` is 2 words; it need not be aligned. */
uint64_t SipHashC(const uint64_t* key, const char* bytes, const uint64_t size);
uint64_t SipHash13C(const uint64_t* key, const char* bytes,
                    const uint64_t size);

/* 64-bit HighwayHash on the best implementation for the current CPU. `key` is
 * 4 words; it need not be aligned. `bytes` may be NULL if `size` is 0. */
uint64_t HighwayHash64(const uint64_t* key, const char* bytes,
                       const uint64_t size);

/* The same on one implementation. Unlike upstream, a target the CPU (or the
 * build) lacks gives the portable result instead of crashing. There is no VSX
 * implementation: _TargetVSX is the portable code. */
uint64_t HighwayHash64_TargetPortable(const uint64_t* key, const char* bytes,
                                      const uint64_t size);
uint64_t HighwayHash64_TargetSSE41(const uint64_t* key, const char* bytes,
                                   const uint64_t size);
uint64_t HighwayHash64_TargetAVX2(const uint64_t* key, const char* bytes,
                                  const uint64_t size);
uint64_t HighwayHash64_TargetNEON(const uint64_t* key, const char* bytes,
                                  const uint64_t size);
uint64_t HighwayHash64_TargetVSX(const uint64_t* key, const char* bytes,
                                 const uint64_t size);

#ifdef __cplusplus
}
#endif

#endif /* FAIRWAY_CAPI_H_ */
