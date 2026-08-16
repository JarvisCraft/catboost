#pragma once

// Lets avx512_kernels.h be compiled on a host without AVX-512, by routing the
// intrinsics through SIMDe and standing in for the few util/ names the kernels
// use. Only the kernel check includes this; the shipped evaluator always uses
// the real intrinsics.
//
// SIMDe 0.8.2 does not implement every intrinsic the kernels need, so the
// missing five are emulated here with plain loops. They are only ever used to
// verify results, never to measure anything.

#define SIMDE_ENABLE_NATIVE_ALIASES 1

#include <simde/x86/avx512.h>
#include <simde/x86/avx2.h>

#include <cstddef>
#include <cstdint>

// SIMDe aliases __mmask16/32/64 but not __mmask8.
#if !defined(__mmask8)
using __mmask8 = simde__mmask8;
#endif

static inline simde__m512i CbSimdeMaskzLoaduEpi8(simde__mmask64 mask, const void* ptr) {
    alignas(64) std::uint8_t buffer[64] = {};
    const auto* source = static_cast<const std::uint8_t*>(ptr);
    for (int i = 0; i < 64; ++i) {
        if ((mask >> i) & 1) {
            buffer[i] = source[i];
        }
    }
    return simde_mm512_loadu_si512(buffer);
}

static inline void CbSimdeMaskStoreuEpi8(void* ptr, simde__mmask64 mask, simde__m512i value) {
    alignas(64) std::uint8_t buffer[64];
    simde_mm512_storeu_si512(buffer, value);
    auto* destination = static_cast<std::uint8_t*>(ptr);
    for (int i = 0; i < 64; ++i) {
        if ((mask >> i) & 1) {
            destination[i] = buffer[i];
        }
    }
}

static inline simde__m128i CbSimdeMaskzLoaduEpi8128(simde__mmask16 mask, const void* ptr) {
    alignas(16) std::uint8_t buffer[16] = {};
    const auto* source = static_cast<const std::uint8_t*>(ptr);
    for (int i = 0; i < 16; ++i) {
        if ((mask >> i) & 1) {
            buffer[i] = source[i];
        }
    }
    return simde_mm_loadu_si128(reinterpret_cast<const simde__m128i*>(buffer));
}

static inline simde__m512d CbSimdeMaskI32GatherPd(
    simde__m512d src,
    simde__mmask8 mask,
    simde__m256i vindex,
    const void* base,
    int scale)
{
    alignas(32) std::int32_t indexes[8];
    simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(indexes), vindex);
    alignas(64) double values[8];
    simde_mm512_storeu_pd(values, src);
    const auto* bytes = static_cast<const char*>(base);
    for (int i = 0; i < 8; ++i) {
        if ((mask >> i) & 1) {
            double value;
            __builtin_memcpy(&value, bytes + static_cast<std::ptrdiff_t>(indexes[i]) * scale, sizeof(value));
            values[i] = value;
        }
    }
    return simde_mm512_loadu_pd(values);
}

static inline simde__m512d CbSimdeI32GatherPd(simde__m256i vindex, const void* base, int scale) {
    return CbSimdeMaskI32GatherPd(simde_mm512_setzero_pd(), simde__mmask8(0xff), vindex, base, scale);
}

// SIMDe implements the masked loads and stores as a full-width memory access
// followed by a blend. Hardware does not touch the masked-off lanes, and the
// kernels rely on that to run off the end of a document block, so these have to
// be replaced with faithful element-wise versions -- otherwise the check reports
// out-of-bounds accesses that cannot happen with real AVX-512.
static inline simde__m512d CbSimdeMaskzLoaduPd(simde__mmask8 mask, const void* ptr) {
    alignas(64) double buffer[8] = {};
    const auto* source = static_cast<const double*>(ptr);
    for (int i = 0; i < 8; ++i) {
        if ((mask >> i) & 1) {
            buffer[i] = source[i];
        }
    }
    return simde_mm512_loadu_pd(buffer);
}

static inline void CbSimdeMaskStoreuPd(void* ptr, simde__mmask8 mask, simde__m512d value) {
    alignas(64) double buffer[8];
    simde_mm512_storeu_pd(buffer, value);
    auto* destination = static_cast<double*>(ptr);
    for (int i = 0; i < 8; ++i) {
        if ((mask >> i) & 1) {
            destination[i] = buffer[i];
        }
    }
}

static inline simde__m512 CbSimdeMaskzLoaduPs(simde__mmask16 mask, const void* ptr) {
    alignas(64) float buffer[16] = {};
    const auto* source = static_cast<const float*>(ptr);
    for (int i = 0; i < 16; ++i) {
        if ((mask >> i) & 1) {
            buffer[i] = source[i];
        }
    }
    return simde_mm512_loadu_ps(buffer);
}

#undef _mm512_maskz_loadu_pd
#undef _mm512_mask_storeu_pd
#undef _mm512_maskz_loadu_ps
#define _mm512_maskz_loadu_pd(mask, ptr) CbSimdeMaskzLoaduPd((mask), (ptr))
#define _mm512_mask_storeu_pd(ptr, mask, value) CbSimdeMaskStoreuPd((ptr), (mask), (value))
#define _mm512_maskz_loadu_ps(mask, ptr) CbSimdeMaskzLoaduPs((mask), (ptr))

// SIMDe exposes the comparison predicates under its own prefix only.
#ifndef _CMP_GT_OQ
    #define _CMP_GT_OQ SIMDE_CMP_GT_OQ
#endif
#ifndef _CMP_UNORD_Q
    #define _CMP_UNORD_Q SIMDE_CMP_UNORD_Q
#endif

// Intrinsics SIMDe 0.8.2 does not have at all, emulated element by element.
template <class T>
static inline T CbSimdeReduceAdd(const T* values, int count) {
    T sum = T(0);
    for (int i = 0; i < count; ++i) {
        sum += values[i];
    }
    return sum;
}

static inline float CbSimdeReduceAddPs(simde__m512 v) {
    alignas(64) float values[16];
    simde_mm512_storeu_ps(values, v);
    return CbSimdeReduceAdd(values, 16);
}

static inline double CbSimdeReduceAddPd(simde__m512d v) {
    alignas(64) double values[8];
    simde_mm512_storeu_pd(values, v);
    return CbSimdeReduceAdd(values, 8);
}

static inline std::int32_t CbSimdeReduceAddEpi32(simde__m512i v) {
    alignas(64) std::int32_t values[16];
    simde_mm512_storeu_si512(values, v);
    return CbSimdeReduceAdd(values, 16);
}

static inline std::int64_t CbSimdeReduceAddEpi64(simde__m512i v) {
    alignas(64) std::int64_t values[8];
    simde_mm512_storeu_si512(values, v);
    return CbSimdeReduceAdd(values, 8);
}

static inline simde__m512i CbSimdeCvtepu8Epi16(simde__m256i v) {
    alignas(32) std::uint8_t source[32];
    simde_mm256_storeu_si256(reinterpret_cast<simde__m256i*>(source), v);
    alignas(64) std::int16_t widened[32];
    for (int i = 0; i < 32; ++i) {
        widened[i] = std::int16_t(source[i]);
    }
    return simde_mm512_loadu_si512(widened);
}

static inline simde__m512i CbSimdeCvtepi8Epi32(simde__m128i v) {
    alignas(16) std::int8_t source[16];
    simde_mm_storeu_si128(reinterpret_cast<simde__m128i*>(source), v);
    alignas(64) std::int32_t widened[16];
    for (int i = 0; i < 16; ++i) {
        widened[i] = std::int32_t(source[i]);
    }
    return simde_mm512_loadu_si512(widened);
}

static inline simde__m512i CbSimdeMaskzLoaduEpi32(simde__mmask16 mask, const void* ptr) {
    alignas(64) std::int32_t buffer[16] = {};
    const auto* source = static_cast<const std::int32_t*>(ptr);
    for (int i = 0; i < 16; ++i) {
        if ((mask >> i) & 1) {
            buffer[i] = source[i];
        }
    }
    return simde_mm512_loadu_si512(buffer);
}

static inline simde__m256i CbSimdeMaskzLoaduEpi8256(simde__mmask32 mask, const void* ptr) {
    alignas(32) std::uint8_t buffer[32] = {};
    const auto* source = static_cast<const std::uint8_t*>(ptr);
    for (int i = 0; i < 32; ++i) {
        if ((mask >> i) & 1) {
            buffer[i] = source[i];
        }
    }
    return simde_mm256_loadu_si256(reinterpret_cast<const simde__m256i*>(buffer));
}

#define _mm512_reduce_add_ps(v) CbSimdeReduceAddPs(v)
#define _mm512_reduce_add_pd(v) CbSimdeReduceAddPd(v)
#define _mm512_reduce_add_epi32(v) CbSimdeReduceAddEpi32(v)
#define _mm512_reduce_add_epi64(v) CbSimdeReduceAddEpi64(v)
#define _mm512_cvtepu8_epi16(v) CbSimdeCvtepu8Epi16(v)
#define _mm512_cvtepi8_epi32(v) CbSimdeCvtepi8Epi32(v)
#define _mm512_maskz_loadu_epi32(mask, ptr) CbSimdeMaskzLoaduEpi32((mask), (ptr))
#define _mm256_maskz_loadu_epi8(mask, ptr) CbSimdeMaskzLoaduEpi8256((mask), (ptr))

#define _mm512_maskz_loadu_epi8(mask, ptr) CbSimdeMaskzLoaduEpi8((mask), (ptr))
#define _mm512_mask_storeu_epi8(ptr, mask, value) CbSimdeMaskStoreuEpi8((ptr), (mask), (value))
#define _mm_maskz_loadu_epi8(mask, ptr) CbSimdeMaskzLoaduEpi8128((mask), (ptr))
#define _mm512_i32gather_pd(vindex, base, scale) CbSimdeI32GatherPd((vindex), (base), (scale))
#define _mm512_mask_i32gather_pd(src, mask, vindex, base, scale) \
    CbSimdeMaskI32GatherPd((src), (mask), (vindex), (base), (scale))
