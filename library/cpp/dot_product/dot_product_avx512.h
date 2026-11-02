#pragma once

#include "common.h"

#include <util/system/types.h>
#include <util/system/compiler.h>

#include <stddef.h>

// 512-bit variants of the entry points where doubling the vector width is a
// straight win: floating point and 32-bit integers. The int8 paths already have
// an AVX-512 implementation in dot_product_vnni.h and are left to it.
//
// Each of these falls back to the AVX2 implementation for lengths too short to
// fill a 512-bit register.

Y_PURE_FUNCTION
ui32 DotProductAvx512(const ui8* lhs, const ui8* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
i64 DotProductAvx512(const i32* lhs, const i32* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
float DotProductAvx512(const float* lhs, const float* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
double DotProductAvx512(const double* lhs, const double* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
float DotProductFloatI8Avx512(const float* lhs, const i8* rhs, size_t length) noexcept;

Y_PURE_FUNCTION
TTriWayDotProduct<float> TriWayDotProductAvx512
    (const float* lhs, const float* rhs, size_t length, bool computeRR) noexcept;

TTriWayDotProductFloatI8 TriWayDotProductFloatI8Avx512(
    const float* lhs,
    const i8* rhs,
    size_t length) noexcept;
