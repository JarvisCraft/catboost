// AVX-512 evaluator for oblivious trees, and the AVX-512 binarizer.
//
// Built with -mavx512f -mavx512bw -mavx512dq -mavx512vl (see the `.avx512`
// source group in the generated CMakeLists), so nothing in here may run before
// HaveAvx512Evaluator() has said yes. That check itself lives in a baseline
// translation unit (quantization.cpp) for exactly this reason.
//
// The structure mirrors CalcTreesBlockedImpl() in evaluator_impl.cpp; what
// differs is the register width (64 documents instead of 16), the absence of a
// scalar epilogue (masked stores cover the remainder), the absence of the
// result-buffer alignment dance (AVX-512 loads and stores are unaligned here)
// and the use of vgatherdpd for the leaf values.

#include "avx512_dispatch.h"
#include "avx512_kernels.h"
#include "evaluator.h"
#include "evaluator_common.h"

#include <util/generic/algorithm.h>

#include <cstring>

namespace NCB::NModelEvaluation {

    using namespace NAvx512;

    void BinarizeFloatsAvx512(
        size_t docCount,
        const float* values,
        TConstArrayRef<float> borders,
        ui8*& result,
        bool useNanSubstitution,
        float nanSubstitutionValue)
    {
        if (useNanSubstitution) {
            BinarizeFloatsAvx512Impl<true>(
                docCount, values, borders.data(), borders.size(), MAX_VALUES_PER_BIN, result, nanSubstitutionValue);
        } else {
            BinarizeFloatsAvx512Impl<false>(
                docCount, values, borders.data(), borders.size(), MAX_VALUES_PER_BIN, result, nanSubstitutionValue);
        }
        result += docCount * ((borders.size() + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN);
    }

    template <bool IsSingleClassModel, bool NeedXorMask, size_t BlockCount, bool CalcLeafIndexesOnly>
    Y_FORCE_INLINE void CalcTreesBlockedAvx512Impl(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const ui8* __restrict binFeatures,
        const size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVecUI32,
        size_t treeStart,
        const size_t treeEnd,
        double* __restrict resultsPtr)
    {
        const TRepackedBin* __restrict treeSplitsCurPtr =
            trees.GetRepackedBins().data() + trees.GetModelTreeData()->GetTreeStartOffsets()[treeStart];

        ui8* __restrict indexesVec = (ui8*)indexesVecUI32;
        const double* __restrict treeLeafPtr = trees.GetModelTreeData()->GetLeafValues().data();
        const size_t* __restrict firstLeafOffsetsPtr = applyData.TreeFirstLeafOffsets.data();
        const auto& treeSizes = trees.GetModelTreeData()->GetTreeSizes();

        if constexpr (IsSingleClassModel && !CalcLeafIndexesOnly) {
            // Trees deeper than 8 do not fit a one-byte leaf index, so the whole
            // group has to fall back if any of them is deep.
            const bool allTreesAreShallow = AllOf(
                treeSizes.begin() + treeStart,
                treeSizes.begin() + treeEnd,
                [](int depth) { return depth <= 8; }
            );
            if (allTreesAreShallow) {
                const size_t treeEnd4 = treeStart + (((treeEnd - treeStart) | 0x3) ^ 0x3);
                for (size_t treeId = treeStart; treeId < treeEnd4; treeId += 4) {
                    // The kernel writes every document of the block, so unlike
                    // the SSE path there is nothing to zero out first.
                    for (size_t subTreeId = 0; subTreeId < 4; ++subTreeId) {
                        CalcIndexesAvx512<NeedXorMask, BlockCount>(
                            binFeatures,
                            docCountInBlock,
                            indexesVec + docCountInBlock * subTreeId,
                            treeSplitsCurPtr,
                            treeSizes[treeId + subTreeId]);
                        treeSplitsCurPtr += treeSizes[treeId + subTreeId];
                    }
                    CalculateLeafValues4Avx512(
                        docCountInBlock,
                        treeLeafPtr + firstLeafOffsetsPtr[treeId + 0],
                        treeLeafPtr + firstLeafOffsetsPtr[treeId + 1],
                        treeLeafPtr + firstLeafOffsetsPtr[treeId + 2],
                        treeLeafPtr + firstLeafOffsetsPtr[treeId + 3],
                        indexesVec + docCountInBlock * 0,
                        indexesVec + docCountInBlock * 1,
                        indexesVec + docCountInBlock * 2,
                        indexesVec + docCountInBlock * 3,
                        resultsPtr
                    );
                }
                treeStart = treeEnd4;
            }
        }

        for (size_t treeId = treeStart; treeId < treeEnd; ++treeId) {
            const auto curTreeSize = treeSizes[treeId];
            if constexpr (!CalcLeafIndexesOnly) {
                if (curTreeSize <= 8) {
                    CalcIndexesAvx512<NeedXorMask, BlockCount>(
                        binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr, curTreeSize);
                    if constexpr (IsSingleClassModel) {
                        CalculateLeafValuesAvx512(
                            docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVec, resultsPtr);
                    } else {
                        CalculateLeafValuesMulti(
                            docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVec,
                            trees.GetDimensionsCount(), resultsPtr);
                    }
                    treeSplitsCurPtr += curTreeSize;
                    continue;
                }
            }
            // Deep trees and leaf-index extraction need the 32-bit index array,
            // which CalcIndexesBasic() accumulates into rather than overwrites.
            memset(indexesVecUI32, 0, sizeof(ui32) * docCountInBlock);
            CalcIndexesBasic<NeedXorMask>(
                binFeatures, docCountInBlock, indexesVecUI32, treeSplitsCurPtr, curTreeSize);
            if constexpr (CalcLeafIndexesOnly) {
                indexesVecUI32 += docCountInBlock;
                indexesVec += sizeof(ui32) * docCountInBlock;
            } else if constexpr (IsSingleClassModel) {
                CalculateLeafValues(
                    docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVecUI32, resultsPtr);
            } else {
                CalculateLeafValuesMulti(
                    docCountInBlock, treeLeafPtr + firstLeafOffsetsPtr[treeId], indexesVecUI32,
                    trees.GetDimensionsCount(), resultsPtr);
            }
            treeSplitsCurPtr += curTreeSize;
        }
    }

    template <bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly>
    void CalcTreesBlockedAvx512(
        const TModelTrees& trees,
        const TModelTrees::TForApplyData& applyData,
        const TCPUEvaluatorQuantizedData* quantizedData,
        size_t docCountInBlock,
        TCalcerIndexType* __restrict indexesVec,
        size_t treeStart,
        size_t treeEnd,
        double* __restrict resultsPtr)
    {
        const ui8* __restrict binFeatures = quantizedData->QuantizedData.data();

#define CALC_TREES_BLOCKED_AVX512_CASE(blockCount)                                                    \
    case blockCount:                                                                                  \
        CalcTreesBlockedAvx512Impl<IsSingleClassModel, NeedXorMask, blockCount, CalcLeafIndexesOnly>( \
            trees, applyData, binFeatures, docCountInBlock, indexesVec, treeStart, treeEnd, resultsPtr); \
        break;

        switch (docCountInBlock / AVX512_BLOCK_SIZE) {
            CALC_TREES_BLOCKED_AVX512_CASE(0)
            CALC_TREES_BLOCKED_AVX512_CASE(1)
            CALC_TREES_BLOCKED_AVX512_CASE(2)
            CALC_TREES_BLOCKED_AVX512_CASE(3)
            CALC_TREES_BLOCKED_AVX512_CASE(4)
            CALC_TREES_BLOCKED_AVX512_CASE(5)
            CALC_TREES_BLOCKED_AVX512_CASE(6)
            CALC_TREES_BLOCKED_AVX512_CASE(7)
            CALC_TREES_BLOCKED_AVX512_CASE(8)
            default:
                CB_ENSURE(false, "Unexpected number of AVX-512 blocks");
        }
#undef CALC_TREES_BLOCKED_AVX512_CASE
    }

    template <bool IsSingleClassModel, bool NeedXorMask, bool CalcLeafIndexesOnly>
    struct TCalcTreesBlockedAvx512Getter {
        TTreeCalcFunction operator()() const {
            return CalcTreesBlockedAvx512<IsSingleClassModel, NeedXorMask, CalcLeafIndexesOnly>;
        }
    };

    TTreeCalcFunction GetCalcTreesFunctionAvx512(
        const TModelTrees& trees,
        size_t docCountInBlock,
        bool calcIndexesOnly
    ) {
        // Non-oblivious trees and the single-document path have no AVX-512
        // variant; an empty function tells the caller to keep the baseline one.
        if (!trees.IsOblivious() || docCountInBlock <= 1) {
            return {};
        }
        if (docCountInBlock > AVX512_BLOCK_SIZE * MAX_AVX512_BLOCK_COUNT) {
            return {};
        }
        const bool isSingleClassModel = (trees.GetDimensionsCount() == 1);
        const bool needXorMask = !trees.GetOneHotFeatures().empty();
        return FunctorTemplateParamsSubstitutor<TCalcTreesBlockedAvx512Getter>::Call(
            isSingleClassModel, needXorMask, calcIndexesOnly);
    }
}
