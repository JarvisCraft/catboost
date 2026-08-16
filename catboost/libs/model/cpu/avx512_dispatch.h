#pragma once

// Entry points into the AVX-512 evaluator that the baseline (SSE-compiled)
// translation units are allowed to see. The implementations live in
// evaluator_impl_avx512.cpp, which is built with -mavx512*, so nothing declared
// here may be inlined into a baseline caller.

#include <util/generic/array_ref.h>
#include <util/system/platform.h>
#include <util/system/types.h>

#include <cstddef>

namespace NCB::NModelEvaluation {

    // Document block size the AVX-512 evaluator asks for when the caller can
    // hand it a transposed input matrix. The paper this work follows measures
    // the transposed layout getting monotonically faster with larger blocks --
    // one feature row of a block then stays contiguous, so walking features does
    // not keep jumping over other blocks -- while the non-transposed layout
    // peaks at 128 and degrades past it.
    constexpr size_t AVX512_TRANSPOSED_BLOCK_SIZE = 512;

    // True when the CPU has AVX-512F/BW/DQ/VL and the CATBOOST_NO_AVX512
    // environment variable is unset. The answer is computed once.
    bool HaveAvx512Evaluator();

    // AVX-512 quantile binarization of one float feature, reading the documents
    // straight from memory instead of through a per-document accessor -- only
    // the transposed input layout can offer that. Writes
    // ceil(borders.size() / MAX_VALUES_PER_BIN) rows of `docCount` bytes and
    // advances `result` past them, like BinarizeFloats() does.
    void BinarizeFloatsAvx512(
        size_t docCount,
        const float* values,
        TConstArrayRef<float> borders,
        ui8*& result,
        bool useNanSubstitution,
        float nanSubstitutionValue);
}
