#pragma once

// Stand-ins for the AVX2 dot products that dot_product_avx512.cpp falls back to
// for short inputs. The standalone check is about the AVX-512 code, so these are
// plain loops -- what they verify is that the dispatch hands short inputs over
// correctly, not the AVX2 implementations themselves.

// Included before the definitions so that the attributes on the real
// declarations are seen first.
#include "../../library/cpp/dot_product/dot_product_avx2.h"

inline i32 DotProductAvx2(const i8* lhs, const i8* rhs, size_t length) noexcept {
    i32 sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += i32(lhs[i]) * i32(rhs[i]);
    }
    return sum;
}

inline ui32 DotProductAvx2(const ui8* lhs, const ui8* rhs, size_t length) noexcept {
    ui32 sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += ui32(lhs[i]) * ui32(rhs[i]);
    }
    return sum;
}

inline i64 DotProductAvx2(const i32* lhs, const i32* rhs, size_t length) noexcept {
    i64 sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += i64(lhs[i]) * i64(rhs[i]);
    }
    return sum;
}

inline float DotProductAvx2(const float* lhs, const float* rhs, size_t length) noexcept {
    float sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

inline double DotProductAvx2(const double* lhs, const double* rhs, size_t length) noexcept {
    double sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

inline float DotProductFloatI8Avx2(const float* lhs, const i8* rhs, size_t length) noexcept {
    float sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += lhs[i] * float(rhs[i]);
    }
    return sum;
}

inline TTriWayDotProduct<float> TriWayDotProductAvx2(
    const float* lhs, const float* rhs, size_t length, bool computeRR) noexcept
{
    TTriWayDotProduct<float> result{};
    result.LL = 0;
    result.LR = 0;
    float rr = 0;
    for (size_t i = 0; i < length; ++i) {
        result.LL += lhs[i] * lhs[i];
        result.LR += lhs[i] * rhs[i];
        rr += rhs[i] * rhs[i];
    }
    if (computeRR) {
        result.RR = rr;
    }
    return result;
}

inline TTriWayDotProductFloatI8 TriWayDotProductFloatI8Avx2(
    const float* lhs, const i8* rhs, size_t length) noexcept
{
    TTriWayDotProductFloatI8 result{};
    for (size_t i = 0; i < length; ++i) {
        result.LL += lhs[i] * lhs[i];
        result.LR += lhs[i] * float(rhs[i]);
        result.RR += float(rhs[i]) * float(rhs[i]);
    }
    return result;
}
