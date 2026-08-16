#pragma once

// AVX-512 kernels for the CPU evaluator: quantile binarization, tree index
// calculation and leaf value fetching.
//
// Everything here is pure intrinsics parameterized by plain types, so the file
// can only be included from a translation unit compiled with
// -mavx512f -mavx512bw -mavx512dq -mavx512vl (see the `.avx512` source group in
// the generated CMakeLists) or from the standalone kernel test, which maps the
// intrinsics onto SIMDe to check them on a host without AVX-512.
//
// Register width is 64 bytes, so all kernels walk documents in groups of 64
// (bytes) or 8 (doubles). Remainders are handled by the very same code under a
// write mask instead of a scalar epilogue: an under-filled vector iteration is
// cheaper than a handful of scalar ones.

#ifdef CATBOOST_AVX512_KERNELS_WITH_SIMDE
    // Supplies the intrinsics through SIMDe. The kernel check also puts stand-ins
    // for the util/ headers below on the include path.
    #include "avx512_simde_compat.h"
#else
    #include <immintrin.h>
#endif

#include <util/generic/utility.h>
#include <util/system/compiler.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <cstddef>

namespace NCB::NModelEvaluation::NAvx512 {

    // Documents per 512-bit register of quantized (one byte per document) data.
    constexpr size_t AVX512_BLOCK_SIZE = 64;

    // Largest document block the kernels below are instantiated for.
    constexpr size_t MAX_AVX512_BLOCK_COUNT = 8;

    Y_FORCE_INLINE __mmask64 MakeTailMask64(size_t count) {
        Y_ASSERT(count < 64);
        return (__mmask64(1) << count) - 1;
    }

    Y_FORCE_INLINE __mmask16 MakeTailMask16(size_t count) {
        Y_ASSERT(count <= 16);
        return __mmask16((ui32(1) << count) - 1);
    }

    Y_FORCE_INLINE __mmask8 MakeTailMask8(size_t count) {
        Y_ASSERT(count <= 8);
        return __mmask8((ui32(1) << count) - 1);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Index calculation
    ////////////////////////////////////////////////////////////////////////////

    // One depth level for one register worth of documents: sets bit `depth` of
    // every document whose quantized feature value is at or above the split
    // border. `TSplit` only has to expose FeatureIndex, SplitIdx and XorMask,
    // which lets the kernel test pass its own split description.
    template <bool NeedXorMask, class TSplit>
    Y_FORCE_INLINE __m512i UpdateIndexesRegister(
        __m512i indexes,
        __m512i binValues,
        const TSplit& split,
        int depth)
    {
        if constexpr (NeedXorMask) {
            binValues = _mm512_xor_si512(binValues, _mm512_set1_epi8((char)split.XorMask));
        }
        const __mmask64 goesRight = _mm512_cmpge_epu8_mask(binValues, _mm512_set1_epi8((char)split.SplitIdx));
        return _mm512_or_si512(indexes, _mm512_maskz_set1_epi8(goesRight, (char)(1 << depth)));
    }

    // Calculates leaf indexes of one oblivious tree for a block of documents.
    // `BlockCount` full registers are kept in flight at once (the depth loop is
    // the outer one) so that the per-depth broadcast of the border is shared by
    // all of them; the remaining `docCountInBlock % 64` documents are done by a
    // masked iteration.
    //
    // `binFeatures` is feature-major: feature `f` of document `d` lives at
    // binFeatures[f * docCountInBlock + d]. `indexesVec` must have room for
    // `docCountInBlock` bytes.
    template <bool NeedXorMask, size_t BlockCount, int CurTreeSize, class TSplit>
    Y_FORCE_INLINE void CalcIndexesAvx512Depthed(
        const ui8* __restrict binFeatures,
        size_t docCountInBlock,
        ui8* __restrict indexesVec,
        const TSplit* __restrict treeSplitsCurPtr)
    {
        const size_t tailSize = docCountInBlock - BlockCount * AVX512_BLOCK_SIZE;
        const __mmask64 tailMask = tailSize == 0 ? __mmask64(0) : MakeTailMask64(tailSize);

        __m512i accumulators[BlockCount == 0 ? 1 : BlockCount];
        for (size_t regId = 0; regId < BlockCount; ++regId) {
            accumulators[regId] = _mm512_setzero_si512();
        }
        __m512i tailAccumulator = _mm512_setzero_si512();

        for (int depth = 0; depth < CurTreeSize; ++depth) {
            const TSplit& split = treeSplitsCurPtr[depth];
            const ui8* __restrict binFeaturePtr = binFeatures + split.FeatureIndex * docCountInBlock;
            for (size_t regId = 0; regId < BlockCount; ++regId) {
                const __m512i binValues =
                    _mm512_loadu_si512((const __m512i*)(binFeaturePtr + AVX512_BLOCK_SIZE * regId));
                accumulators[regId] =
                    UpdateIndexesRegister<NeedXorMask>(accumulators[regId], binValues, split, depth);
            }
            if (tailMask != 0) {
                const __m512i binValues =
                    _mm512_maskz_loadu_epi8(tailMask, binFeaturePtr + AVX512_BLOCK_SIZE * BlockCount);
                tailAccumulator = UpdateIndexesRegister<NeedXorMask>(tailAccumulator, binValues, split, depth);
            }
        }

        for (size_t regId = 0; regId < BlockCount; ++regId) {
            _mm512_storeu_si512((__m512i*)(indexesVec + AVX512_BLOCK_SIZE * regId), accumulators[regId]);
        }
        if (tailMask != 0) {
            _mm512_mask_storeu_epi8(indexesVec + AVX512_BLOCK_SIZE * BlockCount, tailMask, tailAccumulator);
        }
    }

    template <bool NeedXorMask, size_t BlockCount, class TSplit>
    Y_FORCE_INLINE void CalcIndexesAvx512(
        const ui8* __restrict binFeatures,
        size_t docCountInBlock,
        ui8* __restrict indexesVec,
        const TSplit* __restrict treeSplitsCurPtr,
        int curTreeSize)
    {
        switch (curTreeSize) {
            case 1:
                CalcIndexesAvx512Depthed<NeedXorMask, BlockCount, 1>(
                    binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
                break;
            case 2:
                CalcIndexesAvx512Depthed<NeedXorMask, BlockCount, 2>(
                    binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
                break;
            case 3:
                CalcIndexesAvx512Depthed<NeedXorMask, BlockCount, 3>(
                    binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
                break;
            case 4:
                CalcIndexesAvx512Depthed<NeedXorMask, BlockCount, 4>(
                    binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
                break;
            case 5:
                CalcIndexesAvx512Depthed<NeedXorMask, BlockCount, 5>(
                    binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
                break;
            case 6:
                CalcIndexesAvx512Depthed<NeedXorMask, BlockCount, 6>(
                    binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
                break;
            case 7:
                CalcIndexesAvx512Depthed<NeedXorMask, BlockCount, 7>(
                    binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
                break;
            case 8:
                CalcIndexesAvx512Depthed<NeedXorMask, BlockCount, 8>(
                    binFeatures, docCountInBlock, indexesVec, treeSplitsCurPtr);
                break;
            default:
                break;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Leaf values
    ////////////////////////////////////////////////////////////////////////////

    // Eight leaf values picked by eight one-byte indexes. `vgatherdpd` takes
    // 32-bit offsets, so the indexes are zero-extended first.
    Y_FORCE_INLINE __m512d GatherLeaves(const double* __restrict treeLeafPtr, const ui8* __restrict indexesPtr) {
        const __m256i indexes = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)indexesPtr));
        return _mm512_i32gather_pd(indexes, treeLeafPtr, sizeof(double));
    }

    Y_FORCE_INLINE __m512d GatherLeavesMasked(
        const double* __restrict treeLeafPtr,
        const ui8* __restrict indexesPtr,
        __mmask8 mask)
    {
        const __m256i indexes = _mm256_cvtepu8_epi32(_mm_maskz_loadu_epi8((__mmask16)mask, indexesPtr));
        return _mm512_mask_i32gather_pd(_mm512_setzero_pd(), mask, indexes, treeLeafPtr, sizeof(double));
    }

    // writePtr[d] += treeLeafPtr[indexesPtr[d]] for a single tree.
    Y_FORCE_INLINE void CalculateLeafValuesAvx512(
        size_t docCountInBlock,
        const double* __restrict treeLeafPtr,
        const ui8* __restrict indexesPtr,
        double* __restrict writePtr)
    {
        size_t docId = 0;
        for (; docId + 8 <= docCountInBlock; docId += 8) {
            const __m512d sum = _mm512_add_pd(
                _mm512_loadu_pd(writePtr + docId),
                GatherLeaves(treeLeafPtr, indexesPtr + docId));
            _mm512_storeu_pd(writePtr + docId, sum);
        }
        if (docId < docCountInBlock) {
            const __mmask8 mask = MakeTailMask8(docCountInBlock - docId);
            const __m512d sum = _mm512_add_pd(
                _mm512_maskz_loadu_pd(mask, writePtr + docId),
                GatherLeavesMasked(treeLeafPtr, indexesPtr + docId, mask));
            _mm512_mask_storeu_pd(writePtr + docId, mask, sum);
        }
    }

    // Same, for four trees at once: the accumulator round trip through memory is
    // paid once instead of four times.
    Y_FORCE_INLINE void CalculateLeafValues4Avx512(
        size_t docCountInBlock,
        const double* __restrict treeLeafPtr0,
        const double* __restrict treeLeafPtr1,
        const double* __restrict treeLeafPtr2,
        const double* __restrict treeLeafPtr3,
        const ui8* __restrict indexesPtr0,
        const ui8* __restrict indexesPtr1,
        const ui8* __restrict indexesPtr2,
        const ui8* __restrict indexesPtr3,
        double* __restrict writePtr)
    {
        size_t docId = 0;
        for (; docId + 8 <= docCountInBlock; docId += 8) {
            __m512d sum = _mm512_loadu_pd(writePtr + docId);
            sum = _mm512_add_pd(sum, GatherLeaves(treeLeafPtr0, indexesPtr0 + docId));
            sum = _mm512_add_pd(sum, GatherLeaves(treeLeafPtr1, indexesPtr1 + docId));
            sum = _mm512_add_pd(sum, GatherLeaves(treeLeafPtr2, indexesPtr2 + docId));
            sum = _mm512_add_pd(sum, GatherLeaves(treeLeafPtr3, indexesPtr3 + docId));
            _mm512_storeu_pd(writePtr + docId, sum);
        }
        if (docId < docCountInBlock) {
            const __mmask8 mask = MakeTailMask8(docCountInBlock - docId);
            __m512d sum = _mm512_maskz_loadu_pd(mask, writePtr + docId);
            sum = _mm512_add_pd(sum, GatherLeavesMasked(treeLeafPtr0, indexesPtr0 + docId, mask));
            sum = _mm512_add_pd(sum, GatherLeavesMasked(treeLeafPtr1, indexesPtr1 + docId, mask));
            sum = _mm512_add_pd(sum, GatherLeavesMasked(treeLeafPtr2, indexesPtr2 + docId, mask));
            sum = _mm512_add_pd(sum, GatherLeavesMasked(treeLeafPtr3, indexesPtr3 + docId, mask));
            _mm512_mask_storeu_pd(writePtr + docId, mask, sum);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Binarization
    ////////////////////////////////////////////////////////////////////////////

    // `_mm512_packs_*` interleave their operands inside every 128-bit lane, so
    // packing four comparison result registers gives the documents back in
    // 4-byte groups permuted as (4 * (n % 4) + n / 4). One `vpermd` puts them
    // back in order, which is what lets the loads below stay sequential.
    Y_FORCE_INLINE __m512i PackComparisonResults(__m512i r0, __m512i r1, __m512i r2, __m512i r3) {
        const __m512i packed = _mm512_packs_epi16(_mm512_packs_epi32(r0, r1), _mm512_packs_epi32(r2, r3));
        const __m512i order = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        return _mm512_permutexvar_epi32(order, packed);
    }

    Y_FORCE_INLINE __m512 SubstituteNans(__m512 values, __m512 substitutionValue) {
        return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(values, values, _CMP_UNORD_Q), values, substitutionValue);
    }

    // Quantizes one float feature of `docCount` consecutive documents against
    // `borders`, writing `ceil(borderCount / maxValuesPerBin)` rows of
    // `docCount` bytes each, row `i` holding the quantile within border block
    // `i`. Mirrors BinarizeFloats() from quantization.h, but reads the feature
    // straight from memory, which only the transposed input layout allows.
    template <bool UseNanSubstitution>
    void BinarizeFloatsAvx512Impl(
        size_t docCount,
        const float* __restrict values,
        const float* __restrict borders,
        size_t borderCount,
        size_t maxValuesPerBin,
        ui8* __restrict result,
        float nanSubstitutionValue)
    {
        const __m512 substitutionValVec = _mm512_set1_ps(nanSubstitutionValue);
        for (size_t docId = 0; docId < docCount; docId += AVX512_BLOCK_SIZE) {
            const size_t docsLeft = docCount - docId;
            const bool isFullRegister = docsLeft >= AVX512_BLOCK_SIZE;
            const __mmask64 storeMask = isFullRegister ? ~__mmask64(0) : MakeTailMask64(docsLeft);

            __m512 floats[4];
            for (size_t part = 0; part < 4; ++part) {
                const float* partPtr = values + docId + 16 * part;
                if (isFullRegister) {
                    floats[part] = _mm512_loadu_ps(partPtr);
                } else {
                    const size_t partOffset = 16 * part;
                    const size_t partCount = docsLeft > partOffset ? Min<size_t>(16, docsLeft - partOffset) : 0;
                    floats[part] = _mm512_maskz_loadu_ps(MakeTailMask16(partCount), partPtr);
                }
                if constexpr (UseNanSubstitution) {
                    floats[part] = SubstituteNans(floats[part], substitutionValVec);
                }
            }

            ui8* __restrict writePtr = result + docId;
            for (size_t blockStart = 0; blockStart < borderCount; blockStart += maxValuesPerBin) {
                const size_t blockEnd = Min(blockStart + maxValuesPerBin, borderCount);
                __m512i resultVec = _mm512_setzero_si512();
                for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                    const __m512 borderVec = _mm512_set1_ps(borders[borderId]);
                    const __m512i r0 =
                        _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(floats[0], borderVec, _CMP_GT_OQ), 1);
                    const __m512i r1 =
                        _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(floats[1], borderVec, _CMP_GT_OQ), 1);
                    const __m512i r2 =
                        _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(floats[2], borderVec, _CMP_GT_OQ), 1);
                    const __m512i r3 =
                        _mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(floats[3], borderVec, _CMP_GT_OQ), 1);
                    resultVec = _mm512_add_epi8(resultVec, PackComparisonResults(r0, r1, r2, r3));
                }
                _mm512_mask_storeu_epi8(writePtr, storeMask, resultVec);
                writePtr += docCount;
            }
        }
    }
}
