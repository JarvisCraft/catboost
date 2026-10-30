#pragma once

// Instruction-set independent parts of the oblivious tree evaluator, shared by
// evaluator_impl.cpp (baseline) and evaluator_impl_avx512.cpp. The latter is
// compiled with AVX-512 flags, so anything living here must stay free of
// intrinsics.

#include "evaluator.h"

#include <util/system/compiler.h>

namespace NCB::NModelEvaluation {

    // Scalar (auto-vectorized) index calculation, used for trees deeper than 8
    // and, in the baseline evaluator, for the documents left over by the SIMD
    // loop. `startDocId` is always a compile-time constant at the call sites, so
    // the skipped prefix costs nothing.
    template <bool NeedXorMask, typename TIndexType>
    Y_FORCE_INLINE void CalcIndexesBasic(
            const ui8* __restrict binFeatures,
            size_t docCountInBlock,
            TIndexType* __restrict indexesVec,
            const TRepackedBin* __restrict treeSplitsCurPtr,
            int curTreeSize,
            size_t startDocId = 0) {
        if (startDocId >= docCountInBlock) {
            return;
        }
        for (int depth = 0; depth < curTreeSize; ++depth) {
            const ui8 borderVal = (ui8)(treeSplitsCurPtr[depth].SplitIdx);

            const auto featureId = treeSplitsCurPtr[depth].FeatureIndex;
            const ui8* __restrict binFeaturePtr = &binFeatures[featureId * docCountInBlock];
            const ui8 xorMask = treeSplitsCurPtr[depth].XorMask;
            if constexpr (NeedXorMask) {
                Y_PREFETCH_READ(binFeaturePtr, 3);
                Y_PREFETCH_WRITE(indexesVec, 3);
                #if defined(__clang__) && !defined(_ubsan_enabled_)
                #pragma clang loop vectorize_width(16)
                #endif
                for (size_t docId = startDocId; docId < docCountInBlock; ++docId) {
                    indexesVec[docId] |= ((binFeaturePtr[docId] ^ xorMask) >= borderVal) << depth;
                }
            } else {
                Y_PREFETCH_READ(binFeaturePtr, 3);
                Y_PREFETCH_WRITE(indexesVec, 3);
                #if defined(__clang__) && !defined(_ubsan_enabled_)
                #pragma clang loop vectorize_width(16)
                #endif
                for (size_t docId = startDocId; docId < docCountInBlock; ++docId) {
                    indexesVec[docId] |= ((binFeaturePtr[docId]) >= borderVal) << depth;
                }
            }
        }
    }

    template <typename TIndexType>
    Y_FORCE_INLINE void CalculateLeafValues(const size_t docCountInBlock, const double* __restrict treeLeafPtr, const TIndexType* __restrict indexesPtr, double* __restrict writePtr) {
        Y_PREFETCH_READ(treeLeafPtr, 3);
        Y_PREFETCH_READ(treeLeafPtr + 128, 3);
        const auto docCountInBlock4 = (docCountInBlock | 0x3) ^ 0x3;
        for (size_t docId = 0; docId < docCountInBlock4; docId += 4) {
            writePtr[0] += treeLeafPtr[indexesPtr[0]];
            writePtr[1] += treeLeafPtr[indexesPtr[1]];
            writePtr[2] += treeLeafPtr[indexesPtr[2]];
            writePtr[3] += treeLeafPtr[indexesPtr[3]];
            writePtr += 4;
            indexesPtr += 4;
        }
        for (size_t docId = docCountInBlock4; docId < docCountInBlock; ++docId) {
            *writePtr += treeLeafPtr[*indexesPtr];
            ++writePtr;
            ++indexesPtr;
        }
    }

    // Turns a run-time list of flags into the matching template instantiation:
    // `Call(a, b, c)` ends up at `TFunctor<a, b, c>()()`.
    template <template <bool...> class TFunctor, bool... params>
    struct FunctorTemplateParamsSubstitutor {
        static auto Call() {
            return TFunctor<params...>()();
        }

        template <typename... Bools>
        static auto Call(bool nextParam, Bools... lastParams) {
            if (nextParam) {
                return FunctorTemplateParamsSubstitutor<TFunctor, params..., true>::Call(lastParams...);
            } else {
                return FunctorTemplateParamsSubstitutor<TFunctor, params..., false>::Call(lastParams...);
            }
        }
    };

    template <typename TIndexType>
    Y_FORCE_INLINE void CalculateLeafValuesMulti(const size_t docCountInBlock, const double* __restrict leafPtr, const TIndexType* __restrict indexesVec, const int approxDimension, double* __restrict writePtr) {
        for (size_t docId = 0; docId < docCountInBlock; ++docId) {
            const double* __restrict leafValuePtr = leafPtr + indexesVec[docId] * approxDimension;
            for (int classId = 0; classId < approxDimension; ++classId) {
                writePtr[classId] += leafValuePtr[classId];
            }
            writePtr += approxDimension;
        }
    }
}
