#include "dot_product_avx512.h"
#include "dot_product_avx2.h"

#include <util/system/compiler.h>
#include <util/system/platform.h>

// CATBOOST_AVX512_KERNELS_WITH_SIMDE lets the standalone kernel check compile
// this file through SIMDe on a host without AVX-512, to verify the results.
#if defined(CATBOOST_AVX512_KERNELS_WITH_SIMDE)
    #define DOT_PRODUCT_AVX512_AVAILABLE
    #include "avx512_simde_compat.h"
#elif defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
    #define DOT_PRODUCT_AVX512_AVAILABLE
    #include <immintrin.h>
#endif

#ifdef DOT_PRODUCT_AVX512_AVAILABLE

namespace {
    // Shortest input worth entering the 512-bit code for: below this the AVX2
    // version does the whole job in one or two registers anyway.
    constexpr size_t MIN_AVX512_LENGTH = 32;

    Y_FORCE_INLINE __mmask16 TailMask16(size_t length) noexcept {
        return __mmask16((ui32(1) << length) - 1);
    }

    Y_FORCE_INLINE __mmask8 TailMask8(size_t length) noexcept {
        return __mmask8((ui32(1) << length) - 1);
    }

    Y_FORCE_INLINE __mmask32 TailMask32(size_t length) noexcept {
        return __mmask32((ui64(1) << length) - 1);
    }

    // Sixteen i8 values as floats, for the float-by-i8 products.
    Y_FORCE_INLINE __m512 LoadFloatI8Rhs16Avx512(const i8* rhs) noexcept {
        return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)rhs)));
    }

    Y_FORCE_INLINE __m512 LoadFloatI8Rhs16MaskedAvx512(const i8* rhs, __mmask16 mask) noexcept {
        return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(mask, rhs)));
    }
}

ui32 DotProductAvx512(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    if (length < MIN_AVX512_LENGTH) {
        return DotProductAvx2(lhs, rhs, length);
    }
    // Widening to 16 bits keeps every product below 2^16, so the signed
    // multiply-add gives the same answer an unsigned one would.
    __m512i sum0 = _mm512_setzero_si512();
    __m512i sum1 = _mm512_setzero_si512();
    while (length >= 64) {
        const __m512i l = _mm512_loadu_si512(lhs);
        const __m512i r = _mm512_loadu_si512(rhs);
        sum0 = _mm512_add_epi32(sum0, _mm512_madd_epi16(
            _mm512_cvtepu8_epi16(_mm512_castsi512_si256(l)),
            _mm512_cvtepu8_epi16(_mm512_castsi512_si256(r))));
        sum1 = _mm512_add_epi32(sum1, _mm512_madd_epi16(
            _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(l, 1)),
            _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(r, 1))));
        lhs += 64;
        rhs += 64;
        length -= 64;
    }
    while (length >= 32) {
        sum0 = _mm512_add_epi32(sum0, _mm512_madd_epi16(
            _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)lhs)),
            _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)rhs))));
        lhs += 32;
        rhs += 32;
        length -= 32;
    }
    if (length > 0) {
        const __mmask32 mask = TailMask32(length);
        sum0 = _mm512_add_epi32(sum0, _mm512_madd_epi16(
            _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, lhs)),
            _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, rhs))));
    }
    return (ui32)_mm512_reduce_add_epi32(_mm512_add_epi32(sum0, sum1));
}

i64 DotProductAvx512(const i32* lhs, const i32* rhs, size_t length) noexcept {
    if (length < MIN_AVX512_LENGTH) {
        return DotProductAvx2(lhs, rhs, length);
    }
    // vpmuldq only multiplies the even 32-bit lanes, so each pass over the data
    // is done twice: once as loaded and once shifted down by one lane.
    __m512i sum = _mm512_setzero_si512();
    while (length >= 16) {
        const __m512i l = _mm512_loadu_si512(lhs);
        const __m512i r = _mm512_loadu_si512(rhs);
        sum = _mm512_add_epi64(sum, _mm512_mul_epi32(l, r));
        sum = _mm512_add_epi64(sum, _mm512_mul_epi32(
            _mm512_srli_epi64(l, 32), _mm512_srli_epi64(r, 32)));
        lhs += 16;
        rhs += 16;
        length -= 16;
    }
    if (length > 0) {
        const __mmask16 mask = TailMask16(length);
        const __m512i l = _mm512_maskz_loadu_epi32(mask, lhs);
        const __m512i r = _mm512_maskz_loadu_epi32(mask, rhs);
        sum = _mm512_add_epi64(sum, _mm512_mul_epi32(l, r));
        sum = _mm512_add_epi64(sum, _mm512_mul_epi32(
            _mm512_srli_epi64(l, 32), _mm512_srli_epi64(r, 32)));
    }
    return _mm512_reduce_add_epi64(sum);
}

float DotProductAvx512(const float* lhs, const float* rhs, size_t length) noexcept {
    if (length < MIN_AVX512_LENGTH) {
        return DotProductAvx2(lhs, rhs, length);
    }
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    while (length >= 32) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs), sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(lhs + 16), _mm512_loadu_ps(rhs + 16), sum1);
        lhs += 32;
        rhs += 32;
        length -= 32;
    }
    if (length >= 16) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs), sum0);
        lhs += 16;
        rhs += 16;
        length -= 16;
    }
    if (length > 0) {
        const __mmask16 mask = TailMask16(length);
        sum0 = _mm512_fmadd_ps(
            _mm512_maskz_loadu_ps(mask, lhs), _mm512_maskz_loadu_ps(mask, rhs), sum0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
}

double DotProductAvx512(const double* lhs, const double* rhs, size_t length) noexcept {
    if (length < MIN_AVX512_LENGTH) {
        return DotProductAvx2(lhs, rhs, length);
    }
    __m512d sum0 = _mm512_setzero_pd();
    __m512d sum1 = _mm512_setzero_pd();
    while (length >= 16) {
        sum0 = _mm512_fmadd_pd(_mm512_loadu_pd(lhs), _mm512_loadu_pd(rhs), sum0);
        sum1 = _mm512_fmadd_pd(_mm512_loadu_pd(lhs + 8), _mm512_loadu_pd(rhs + 8), sum1);
        lhs += 16;
        rhs += 16;
        length -= 16;
    }
    if (length >= 8) {
        sum0 = _mm512_fmadd_pd(_mm512_loadu_pd(lhs), _mm512_loadu_pd(rhs), sum0);
        lhs += 8;
        rhs += 8;
        length -= 8;
    }
    if (length > 0) {
        const __mmask8 mask = TailMask8(length);
        sum0 = _mm512_fmadd_pd(
            _mm512_maskz_loadu_pd(mask, lhs), _mm512_maskz_loadu_pd(mask, rhs), sum0);
    }
    return _mm512_reduce_add_pd(_mm512_add_pd(sum0, sum1));
}

float DotProductFloatI8Avx512(const float* lhs, const i8* rhs, size_t length) noexcept {
    if (length < MIN_AVX512_LENGTH) {
        return DotProductFloatI8Avx2(lhs, rhs, length);
    }
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    while (length >= 32) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(lhs), LoadFloatI8Rhs16Avx512(rhs), sum0);
        sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(lhs + 16), LoadFloatI8Rhs16Avx512(rhs + 16), sum1);
        lhs += 32;
        rhs += 32;
        length -= 32;
    }
    if (length >= 16) {
        sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(lhs), LoadFloatI8Rhs16Avx512(rhs), sum0);
        lhs += 16;
        rhs += 16;
        length -= 16;
    }
    if (length > 0) {
        const __mmask16 mask = TailMask16(length);
        sum0 = _mm512_fmadd_ps(
            _mm512_maskz_loadu_ps(mask, lhs), LoadFloatI8Rhs16MaskedAvx512(rhs, mask), sum0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
}

namespace {
    template <bool ComputeRR>
    Y_FORCE_INLINE void TriWayDotProductIterationAvx512(
        __m512& sumLL, __m512& sumLR, __m512& sumRR, const __m512 l, const __m512 r) noexcept
    {
        sumLL = _mm512_fmadd_ps(l, l, sumLL);
        sumLR = _mm512_fmadd_ps(l, r, sumLR);
        if constexpr (ComputeRR) {
            sumRR = _mm512_fmadd_ps(r, r, sumRR);
        }
    }

    template <bool ComputeRR>
    TTriWayDotProduct<float> TriWayDotProductAvx512Impl(
        const float* lhs,
        const float* rhs,
        size_t length) noexcept
    {
        __m512 sumLL0 = _mm512_setzero_ps();
        __m512 sumLR0 = _mm512_setzero_ps();
        __m512 sumRR0 = _mm512_setzero_ps();
        __m512 sumLL1 = _mm512_setzero_ps();
        __m512 sumLR1 = _mm512_setzero_ps();
        __m512 sumRR1 = _mm512_setzero_ps();

        while (length >= 32) {
            TriWayDotProductIterationAvx512<ComputeRR>(
                sumLL0, sumLR0, sumRR0, _mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
            TriWayDotProductIterationAvx512<ComputeRR>(
                sumLL1, sumLR1, sumRR1, _mm512_loadu_ps(lhs + 16), _mm512_loadu_ps(rhs + 16));
            lhs += 32;
            rhs += 32;
            length -= 32;
        }
        if (length >= 16) {
            TriWayDotProductIterationAvx512<ComputeRR>(
                sumLL0, sumLR0, sumRR0, _mm512_loadu_ps(lhs), _mm512_loadu_ps(rhs));
            lhs += 16;
            rhs += 16;
            length -= 16;
        }
        if (length > 0) {
            const __mmask16 mask = TailMask16(length);
            TriWayDotProductIterationAvx512<ComputeRR>(
                sumLL0, sumLR0, sumRR0,
                _mm512_maskz_loadu_ps(mask, lhs), _mm512_maskz_loadu_ps(mask, rhs));
        }

        TTriWayDotProduct<float> result{};
        result.LL = _mm512_reduce_add_ps(_mm512_add_ps(sumLL0, sumLL1));
        result.LR = _mm512_reduce_add_ps(_mm512_add_ps(sumLR0, sumLR1));
        if constexpr (ComputeRR) {
            result.RR = _mm512_reduce_add_ps(_mm512_add_ps(sumRR0, sumRR1));
        }
        return result;
    }
}

TTriWayDotProduct<float> TriWayDotProductAvx512(
    const float* lhs,
    const float* rhs,
    size_t length,
    bool computeRR) noexcept
{
    if (length < MIN_AVX512_LENGTH) {
        return TriWayDotProductAvx2(lhs, rhs, length, computeRR);
    }
    if (computeRR) {
        return TriWayDotProductAvx512Impl<true>(lhs, rhs, length);
    }
    return TriWayDotProductAvx512Impl<false>(lhs, rhs, length);
}

TTriWayDotProductFloatI8 TriWayDotProductFloatI8Avx512(
    const float* lhs,
    const i8* rhs,
    size_t length) noexcept
{
    if (length < MIN_AVX512_LENGTH) {
        return TriWayDotProductFloatI8Avx2(lhs, rhs, length);
    }
    __m512 sumLL0 = _mm512_setzero_ps();
    __m512 sumLR0 = _mm512_setzero_ps();
    __m512 sumRR0 = _mm512_setzero_ps();
    __m512 sumLL1 = _mm512_setzero_ps();
    __m512 sumLR1 = _mm512_setzero_ps();
    __m512 sumRR1 = _mm512_setzero_ps();

    while (length >= 32) {
        TriWayDotProductIterationAvx512<true>(
            sumLL0, sumLR0, sumRR0, _mm512_loadu_ps(lhs), LoadFloatI8Rhs16Avx512(rhs));
        TriWayDotProductIterationAvx512<true>(
            sumLL1, sumLR1, sumRR1, _mm512_loadu_ps(lhs + 16), LoadFloatI8Rhs16Avx512(rhs + 16));
        lhs += 32;
        rhs += 32;
        length -= 32;
    }
    if (length >= 16) {
        TriWayDotProductIterationAvx512<true>(
            sumLL0, sumLR0, sumRR0, _mm512_loadu_ps(lhs), LoadFloatI8Rhs16Avx512(rhs));
        lhs += 16;
        rhs += 16;
        length -= 16;
    }
    if (length > 0) {
        const __mmask16 mask = TailMask16(length);
        TriWayDotProductIterationAvx512<true>(
            sumLL0, sumLR0, sumRR0,
            _mm512_maskz_loadu_ps(mask, lhs), LoadFloatI8Rhs16MaskedAvx512(rhs, mask));
    }

    TTriWayDotProductFloatI8 result{};
    result.LL = _mm512_reduce_add_ps(_mm512_add_ps(sumLL0, sumLL1));
    result.LR = _mm512_reduce_add_ps(_mm512_add_ps(sumLR0, sumLR1));
    result.RR = _mm512_reduce_add_ps(_mm512_add_ps(sumRR0, sumRR1));
    return result;
}

#else

ui32 DotProductAvx512(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    return DotProductAvx2(lhs, rhs, length);
}

i64 DotProductAvx512(const i32* lhs, const i32* rhs, size_t length) noexcept {
    return DotProductAvx2(lhs, rhs, length);
}

float DotProductAvx512(const float* lhs, const float* rhs, size_t length) noexcept {
    return DotProductAvx2(lhs, rhs, length);
}

double DotProductAvx512(const double* lhs, const double* rhs, size_t length) noexcept {
    return DotProductAvx2(lhs, rhs, length);
}

float DotProductFloatI8Avx512(const float* lhs, const i8* rhs, size_t length) noexcept {
    return DotProductFloatI8Avx2(lhs, rhs, length);
}

TTriWayDotProduct<float> TriWayDotProductAvx512(
    const float* lhs,
    const float* rhs,
    size_t length,
    bool computeRR) noexcept
{
    return TriWayDotProductAvx2(lhs, rhs, length, computeRR);
}

TTriWayDotProductFloatI8 TriWayDotProductFloatI8Avx512(
    const float* lhs,
    const i8* rhs,
    size_t length) noexcept
{
    return TriWayDotProductFloatI8Avx2(lhs, rhs, length);
}

#endif

#undef DOT_PRODUCT_AVX512_AVAILABLE
