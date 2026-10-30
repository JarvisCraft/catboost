#include "quantization.h"

#include <util/system/cpu_id.h>
#include <util/system/env.h>

namespace NCB::NModelEvaluation {

    // Deliberately lives in a baseline translation unit: everything in
    // evaluator_impl_avx512.cpp is compiled with -mavx512*, so the check that
    // decides whether to go there must not itself be built that way.
    bool HaveAvx512Evaluator() {
        static const bool have = [] {
            if (!GetEnv("CATBOOST_NO_AVX512").empty()) {
                return false;
            }
            return NX86::CachedHaveAVX512F()
                && NX86::CachedHaveAVX512BW()
                && NX86::CachedHaveAVX512DQ()
                && NX86::CachedHaveAVX512VL();
        }();
        return have;
    }
}
