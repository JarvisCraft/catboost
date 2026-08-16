// Correctness check for the AVX-512 dot products.
//
// Compiles library/cpp/dot_product/dot_product_avx512.cpp itself -- through
// SIMDe when the host has no AVX-512 -- and compares every entry point against a
// straightforward loop over randomized inputs and every length around the
// 8/16/32/64-element steps the implementations take.
//
// The <32 fallback into the AVX2 implementation is stubbed out with a naive
// version here, so the lengths below 32 check the dispatch rather than AVX2.

#include "dot_product_stubs.h"

#include "../../library/cpp/dot_product/dot_product_avx512.cpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {
    int FailureCount = 0;

    void Report(const std::string& name, bool ok) {
        if (!ok) {
            ++FailureCount;
        }
        std::printf("%-46s %s\n", name.c_str(), ok ? "ok" : "FAILED");
    }

    // Lengths worth trying: everything small, then the boundaries of the 8, 16,
    // 32 and 64 element steps and their neighbours.
    std::vector<size_t> Lengths() {
        std::vector<size_t> lengths;
        for (size_t length = 0; length <= 80; ++length) {
            lengths.push_back(length);
        }
        for (size_t length : {95, 96, 97, 127, 128, 129, 255, 256, 257, 1000}) {
            lengths.push_back(length);
        }
        return lengths;
    }

    bool CloseEnough(double got, double expected, double relativeTolerance) {
        const double scale = std::max(1.0, std::fabs(expected));
        return std::fabs(got - expected) <= relativeTolerance * scale;
    }

    bool CheckFloat(std::mt19937& rng) {
        std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
        for (size_t length : Lengths()) {
            std::vector<float> lhs(length), rhs(length);
            for (size_t i = 0; i < length; ++i) {
                lhs[i] = distribution(rng);
                rhs[i] = distribution(rng);
            }
            double expected = 0.0;
            for (size_t i = 0; i < length; ++i) {
                expected += double(lhs[i]) * double(rhs[i]);
            }
            const float got = DotProductAvx512(lhs.data(), rhs.data(), length);
            // Summation order differs from the reference, so this is float noise
            // scaled by the number of terms.
            if (!CloseEnough(got, expected, 1e-4)) {
                std::printf("  float mismatch at length %zu: %g vs %g\n", length, double(got), expected);
                return false;
            }
        }
        return true;
    }

    bool CheckDouble(std::mt19937& rng) {
        std::uniform_real_distribution<double> distribution(-2.0, 2.0);
        for (size_t length : Lengths()) {
            std::vector<double> lhs(length), rhs(length);
            for (size_t i = 0; i < length; ++i) {
                lhs[i] = distribution(rng);
                rhs[i] = distribution(rng);
            }
            double expected = 0.0;
            for (size_t i = 0; i < length; ++i) {
                expected += lhs[i] * rhs[i];
            }
            const double got = DotProductAvx512(lhs.data(), rhs.data(), length);
            if (!CloseEnough(got, expected, 1e-12)) {
                std::printf("  double mismatch at length %zu: %g vs %g\n", length, got, expected);
                return false;
            }
        }
        return true;
    }

    bool CheckUi8(std::mt19937& rng) {
        for (size_t length : Lengths()) {
            std::vector<ui8> lhs(length), rhs(length);
            for (size_t i = 0; i < length; ++i) {
                lhs[i] = ui8(rng() & 0xff);
                rhs[i] = ui8(rng() & 0xff);
            }
            ui32 expected = 0;
            for (size_t i = 0; i < length; ++i) {
                expected += ui32(lhs[i]) * ui32(rhs[i]);
            }
            const ui32 got = DotProductAvx512(lhs.data(), rhs.data(), length);
            if (got != expected) {
                std::printf("  ui8 mismatch at length %zu: %u vs %u\n", length, got, expected);
                return false;
            }
        }
        return true;
    }

    bool CheckI32(std::mt19937& rng) {
        std::uniform_int_distribution<i32> distribution(
            std::numeric_limits<i32>::min() / 2, std::numeric_limits<i32>::max() / 2);
        for (size_t length : Lengths()) {
            std::vector<i32> lhs(length), rhs(length);
            for (size_t i = 0; i < length; ++i) {
                lhs[i] = distribution(rng);
                rhs[i] = distribution(rng);
            }
            i64 expected = 0;
            for (size_t i = 0; i < length; ++i) {
                expected += i64(lhs[i]) * i64(rhs[i]);
            }
            const i64 got = DotProductAvx512(lhs.data(), rhs.data(), length);
            if (got != expected) {
                std::printf("  i32 mismatch at length %zu: %lld vs %lld\n",
                            length, (long long)got, (long long)expected);
                return false;
            }
        }
        return true;
    }

    bool CheckFloatI8(std::mt19937& rng) {
        std::uniform_real_distribution<float> floatDistribution(-2.0f, 2.0f);
        for (size_t length : Lengths()) {
            std::vector<float> lhs(length);
            std::vector<i8> rhs(length);
            for (size_t i = 0; i < length; ++i) {
                lhs[i] = floatDistribution(rng);
                rhs[i] = i8(rng() & 0xff);
            }
            double expected = 0.0;
            for (size_t i = 0; i < length; ++i) {
                expected += double(lhs[i]) * double(rhs[i]);
            }
            const float got = DotProductFloatI8Avx512(lhs.data(), rhs.data(), length);
            if (!CloseEnough(got, expected, 1e-4)) {
                std::printf("  float/i8 mismatch at length %zu: %g vs %g\n", length, double(got), expected);
                return false;
            }
        }
        return true;
    }

    bool CheckTriWay(std::mt19937& rng) {
        std::uniform_real_distribution<float> distribution(-2.0f, 2.0f);
        for (size_t length : Lengths()) {
            std::vector<float> lhs(length), rhs(length);
            std::vector<i8> rhsI8(length);
            for (size_t i = 0; i < length; ++i) {
                lhs[i] = distribution(rng);
                rhs[i] = distribution(rng);
                rhsI8[i] = i8(rng() & 0xff);
            }
            double expectedLL = 0.0, expectedLR = 0.0, expectedRR = 0.0;
            for (size_t i = 0; i < length; ++i) {
                expectedLL += double(lhs[i]) * double(lhs[i]);
                expectedLR += double(lhs[i]) * double(rhs[i]);
                expectedRR += double(rhs[i]) * double(rhs[i]);
            }

            const auto withRR = TriWayDotProductAvx512(lhs.data(), rhs.data(), length, true);
            if (!CloseEnough(withRR.LL, expectedLL, 1e-4)
                || !CloseEnough(withRR.LR, expectedLR, 1e-4)
                || !CloseEnough(withRR.RR, expectedRR, 1e-4))
            {
                std::printf("  triway mismatch at length %zu\n", length);
                return false;
            }

            // With computeRR off, RR must be left at the type's default.
            const auto withoutRR = TriWayDotProductAvx512(lhs.data(), rhs.data(), length, false);
            const TTriWayDotProduct<float> defaults{};
            if (!CloseEnough(withoutRR.LL, expectedLL, 1e-4)
                || !CloseEnough(withoutRR.LR, expectedLR, 1e-4)
                || withoutRR.RR != defaults.RR)
            {
                std::printf("  triway (no RR) mismatch at length %zu\n", length);
                return false;
            }

            double expectedI8LR = 0.0, expectedI8RR = 0.0;
            for (size_t i = 0; i < length; ++i) {
                expectedI8LR += double(lhs[i]) * double(rhsI8[i]);
                expectedI8RR += double(rhsI8[i]) * double(rhsI8[i]);
            }
            const auto floatI8 = TriWayDotProductFloatI8Avx512(lhs.data(), rhsI8.data(), length);
            if (!CloseEnough(floatI8.LL, expectedLL, 1e-4)
                || !CloseEnough(floatI8.LR, expectedI8LR, 1e-4)
                || !CloseEnough(floatI8.RR, expectedI8RR, 1e-4))
            {
                std::printf("  triway float/i8 mismatch at length %zu\n", length);
                return false;
            }
        }
        return true;
    }
}

int main() {
    std::mt19937 rng(20260816);

#ifdef CATBOOST_AVX512_KERNELS_WITH_SIMDE
    std::printf("mode: SIMDe (algorithm check, no AVX-512 hardware needed)\n");
#else
    std::printf("mode: native AVX-512\n");
#endif

    Report("DotProductAvx512 (float)", CheckFloat(rng));
    Report("DotProductAvx512 (double)", CheckDouble(rng));
    Report("DotProductAvx512 (ui8)", CheckUi8(rng));
    Report("DotProductAvx512 (i32)", CheckI32(rng));
    Report("DotProductFloatI8Avx512", CheckFloatI8(rng));
    Report("TriWayDotProduct*Avx512", CheckTriWay(rng));

    if (FailureCount != 0) {
        std::printf("\n%d check(s) FAILED\n", FailureCount);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
