// Correctness check for the AVX-512 evaluator kernels.
//
// Every kernel in catboost/libs/model/cpu/avx512_kernels.h is compared against a
// straightforward scalar transcription of what the baseline (SSE/non-SSE) code
// in evaluator_impl.cpp and quantization.h computes, over randomized inputs and
// every document count that exercises the masked tails.
//
// Two build modes, see build_and_run.sh:
//   * native  -- real AVX-512 intrinsics, needs an AVX-512 host;
//   * simde   -- the same source through SIMDe, runs anywhere, checks the
//                algorithm (masks, pack/permute order, gather offsets) rather
//                than the hardware.

#include "../../catboost/libs/model/cpu/avx512_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
    constexpr size_t MAX_VALUES_PER_BIN = 254;

    // Same layout as NCatBoost's TRepackedBin; the kernels are templated on it so
    // that this file does not have to pull in catboost/libs/model.
    struct TSplit {
        ui16 FeatureIndex = 0;
        ui8 XorMask = 0;
        ui8 SplitIdx = 0;
    };

    int FailureCount = 0;

    void Report(const std::string& name, bool ok) {
        if (!ok) {
            ++FailureCount;
        }
        std::printf("%-46s %s\n", name.c_str(), ok ? "ok" : "FAILED");
    }

    ////////////////////////////////////////////////////////////////////////////
    // Scalar references
    ////////////////////////////////////////////////////////////////////////////

    template <bool NeedXorMask>
    void CalcIndexesReference(
        const ui8* binFeatures,
        size_t docCountInBlock,
        ui8* indexesVec,
        const TSplit* splits,
        int curTreeSize)
    {
        std::memset(indexesVec, 0, docCountInBlock);
        for (int depth = 0; depth < curTreeSize; ++depth) {
            const ui8 borderVal = splits[depth].SplitIdx;
            const ui8 xorMask = splits[depth].XorMask;
            const ui8* binFeaturePtr = binFeatures + splits[depth].FeatureIndex * docCountInBlock;
            for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                const ui8 value = NeedXorMask ? ui8(binFeaturePtr[docId] ^ xorMask) : binFeaturePtr[docId];
                indexesVec[docId] |= ui8((value >= borderVal) << depth);
            }
        }
    }

    void CalculateLeafValuesReference(
        size_t docCountInBlock,
        const double* treeLeafPtr,
        const ui8* indexesPtr,
        double* writePtr)
    {
        for (size_t docId = 0; docId < docCountInBlock; ++docId) {
            writePtr[docId] += treeLeafPtr[indexesPtr[docId]];
        }
    }

    void BinarizeFloatsReference(
        size_t docCount,
        const float* values,
        const float* borders,
        size_t borderCount,
        ui8* result,
        bool useNanSubstitution,
        float nanSubstitutionValue)
    {
        const size_t rowCount = (borderCount + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
        std::memset(result, 0, rowCount * docCount);
        for (size_t docId = 0; docId < docCount; ++docId) {
            float val = values[docId];
            if (useNanSubstitution && std::isnan(val)) {
                val = nanSubstitutionValue;
            }
            ui8* writePtr = result + docId;
            for (size_t blockStart = 0; blockStart < borderCount; blockStart += MAX_VALUES_PER_BIN) {
                const size_t blockEnd = std::min(blockStart + MAX_VALUES_PER_BIN, borderCount);
                for (size_t borderId = blockStart; borderId < blockEnd; ++borderId) {
                    *writePtr = ui8(*writePtr + ui8(val > borders[borderId]));
                }
                writePtr += docCount;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Checks
    ////////////////////////////////////////////////////////////////////////////

    using namespace NCB::NModelEvaluation::NAvx512;

    // Dispatches on the number of whole registers exactly the way the evaluator
    // does, so the check covers the same instantiations that ship.
    template <bool NeedXorMask>
    void CalcIndexesDispatch(
        const ui8* binFeatures,
        size_t docCountInBlock,
        ui8* indexesVec,
        const TSplit* splits,
        int curTreeSize)
    {
        switch (docCountInBlock / AVX512_BLOCK_SIZE) {
            case 0:
                CalcIndexesAvx512<NeedXorMask, 0>(binFeatures, docCountInBlock, indexesVec, splits, curTreeSize);
                break;
            case 1:
                CalcIndexesAvx512<NeedXorMask, 1>(binFeatures, docCountInBlock, indexesVec, splits, curTreeSize);
                break;
            case 2:
                CalcIndexesAvx512<NeedXorMask, 2>(binFeatures, docCountInBlock, indexesVec, splits, curTreeSize);
                break;
            case 3:
                CalcIndexesAvx512<NeedXorMask, 3>(binFeatures, docCountInBlock, indexesVec, splits, curTreeSize);
                break;
            case 4:
                CalcIndexesAvx512<NeedXorMask, 4>(binFeatures, docCountInBlock, indexesVec, splits, curTreeSize);
                break;
            case 5:
                CalcIndexesAvx512<NeedXorMask, 5>(binFeatures, docCountInBlock, indexesVec, splits, curTreeSize);
                break;
            case 6:
                CalcIndexesAvx512<NeedXorMask, 6>(binFeatures, docCountInBlock, indexesVec, splits, curTreeSize);
                break;
            case 7:
                CalcIndexesAvx512<NeedXorMask, 7>(binFeatures, docCountInBlock, indexesVec, splits, curTreeSize);
                break;
            case 8:
                CalcIndexesAvx512<NeedXorMask, 8>(binFeatures, docCountInBlock, indexesVec, splits, curTreeSize);
                break;
            default:
                std::printf("unexpected block count\n");
                ++FailureCount;
                break;
        }
    }

    template <bool NeedXorMask>
    bool CheckCalcIndexes(std::mt19937& rng) {
        constexpr size_t featureCount = 37;
        for (size_t docCountInBlock = 1; docCountInBlock <= 512; ++docCountInBlock) {
            // A whole span of document counts would be slow to no purpose; the
            // interesting ones are the register boundaries and their neighbours.
            if (docCountInBlock > 8 && docCountInBlock % 64 > 2 && docCountInBlock % 64 < 62) {
                continue;
            }
            std::vector<ui8> binFeatures(featureCount * docCountInBlock);
            for (auto& value : binFeatures) {
                value = ui8(rng() & 0xff);
            }
            for (int curTreeSize = 1; curTreeSize <= 8; ++curTreeSize) {
                std::vector<TSplit> splits(curTreeSize);
                for (auto& split : splits) {
                    split.FeatureIndex = ui16(rng() % featureCount);
                    split.SplitIdx = ui8(rng() & 0xff);
                    split.XorMask = NeedXorMask ? ui8(rng() & 0xff) : ui8(0);
                }

                // Prefilled rather than zeroed: the evaluator relies on the
                // kernel overwriting every document of the block, which is why
                // it does not clear the index buffer first the way the SSE path
                // does. The bytes past the end are the same poison, so an
                // over-wide store is noticed too.
                std::vector<ui8> got(docCountInBlock + 64, 0xcd);
                std::vector<ui8> expected(docCountInBlock);
                CalcIndexesDispatch<NeedXorMask>(
                    binFeatures.data(), docCountInBlock, got.data(), splits.data(), curTreeSize);
                CalcIndexesReference<NeedXorMask>(
                    binFeatures.data(), docCountInBlock, expected.data(), splits.data(), curTreeSize);

                if (!std::equal(expected.begin(), expected.end(), got.begin())) {
                    std::printf("  indexes mismatch: docCount=%zu depth=%d\n", docCountInBlock, curTreeSize);
                    return false;
                }
                for (size_t i = docCountInBlock; i < got.size(); ++i) {
                    if (got[i] != 0xcd) {
                        std::printf("  indexes overwrote guard byte %zu (docCount=%zu)\n", i, docCountInBlock);
                        return false;
                    }
                }
            }
        }
        return true;
    }

    bool CheckLeafValues(std::mt19937& rng) {
        std::uniform_real_distribution<double> leafDistribution(-10.0, 10.0);
        for (size_t docCountInBlock = 1; docCountInBlock <= 512; ++docCountInBlock) {
            if (docCountInBlock > 24 && docCountInBlock % 8 > 1 && docCountInBlock % 8 < 7) {
                continue;
            }
            constexpr size_t leafCount = 256;
            std::vector<std::vector<double>> leaves(4, std::vector<double>(leafCount));
            std::vector<std::vector<ui8>> indexes(4, std::vector<ui8>(docCountInBlock));
            for (size_t treeId = 0; treeId < 4; ++treeId) {
                for (auto& leaf : leaves[treeId]) {
                    leaf = leafDistribution(rng);
                }
                for (auto& index : indexes[treeId]) {
                    index = ui8(rng() & 0xff);
                }
            }

            std::vector<double> initial(docCountInBlock);
            for (auto& value : initial) {
                value = leafDistribution(rng);
            }

            // One tree at a time.
            {
                std::vector<double> got = initial;
                std::vector<double> expected = initial;
                CalculateLeafValuesAvx512(docCountInBlock, leaves[0].data(), indexes[0].data(), got.data());
                CalculateLeafValuesReference(docCountInBlock, leaves[0].data(), indexes[0].data(), expected.data());
                if (got != expected) {
                    std::printf("  single-tree leaf mismatch: docCount=%zu\n", docCountInBlock);
                    return false;
                }
            }

            // Four trees at a time. Summation order matches the kernel's, so the
            // results have to agree bit for bit.
            {
                std::vector<double> got = initial;
                std::vector<double> expected = initial;
                CalculateLeafValues4Avx512(
                    docCountInBlock,
                    leaves[0].data(), leaves[1].data(), leaves[2].data(), leaves[3].data(),
                    indexes[0].data(), indexes[1].data(), indexes[2].data(), indexes[3].data(),
                    got.data());
                for (size_t docId = 0; docId < docCountInBlock; ++docId) {
                    double sum = expected[docId];
                    for (size_t treeId = 0; treeId < 4; ++treeId) {
                        sum += leaves[treeId][indexes[treeId][docId]];
                    }
                    expected[docId] = sum;
                }
                if (got != expected) {
                    std::printf("  four-tree leaf mismatch: docCount=%zu\n", docCountInBlock);
                    return false;
                }
            }
        }
        return true;
    }

    bool CheckBinarizeFloats(std::mt19937& rng) {
        std::uniform_real_distribution<float> valueDistribution(-3.0f, 3.0f);
        const size_t borderCounts[] = {1, 7, 16, 63, 64, 254, 255, 300};
        for (size_t docCount = 1; docCount <= 512; ++docCount) {
            if (docCount > 8 && docCount % 64 > 2 && docCount % 64 < 62) {
                continue;
            }
            for (size_t borderCount : borderCounts) {
                std::vector<float> borders(borderCount);
                for (auto& border : borders) {
                    border = valueDistribution(rng);
                }
                std::sort(borders.begin(), borders.end());

                std::vector<float> values(docCount);
                for (auto& value : values) {
                    // Roughly one document in ten is a NaN, to exercise both the
                    // substituting and the plain instantiation.
                    value = (rng() % 10 == 0) ? std::numeric_limits<float>::quiet_NaN()
                                              : valueDistribution(rng);
                }

                const size_t rowCount = (borderCount + MAX_VALUES_PER_BIN - 1) / MAX_VALUES_PER_BIN;
                const size_t resultSize = rowCount * docCount;
                for (int nanMode = 0; nanMode < 3; ++nanMode) {
                    const bool useNanSubstitution = nanMode != 0;
                    const float substitution = nanMode == 1 ? -std::numeric_limits<float>::infinity()
                                                            : std::numeric_limits<float>::infinity();

                    std::vector<ui8> got(resultSize + 64, 0xcd);
                    std::memset(got.data(), 0, resultSize);
                    std::vector<ui8> expected(resultSize);
                    if (useNanSubstitution) {
                        BinarizeFloatsAvx512Impl<true>(
                            docCount, values.data(), borders.data(), borderCount, MAX_VALUES_PER_BIN,
                            got.data(), substitution);
                    } else {
                        BinarizeFloatsAvx512Impl<false>(
                            docCount, values.data(), borders.data(), borderCount, MAX_VALUES_PER_BIN,
                            got.data(), substitution);
                    }
                    BinarizeFloatsReference(
                        docCount, values.data(), borders.data(), borderCount, expected.data(),
                        useNanSubstitution, substitution);

                    if (!std::equal(expected.begin(), expected.end(), got.begin())) {
                        std::printf(
                            "  binarize mismatch: docCount=%zu borders=%zu nanMode=%d\n",
                            docCount, borderCount, nanMode);
                        return false;
                    }
                    for (size_t i = resultSize; i < got.size(); ++i) {
                        if (got[i] != 0xcd) {
                            std::printf(
                                "  binarize overwrote guard byte %zu (docCount=%zu borders=%zu)\n",
                                i, docCount, borderCount);
                            return false;
                        }
                    }
                }
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

    Report("CalcIndexesAvx512 (no xor mask)", CheckCalcIndexes<false>(rng));
    Report("CalcIndexesAvx512 (xor mask)", CheckCalcIndexes<true>(rng));
    Report("CalculateLeafValues{,4}Avx512", CheckLeafValues(rng));
    Report("BinarizeFloatsAvx512Impl", CheckBinarizeFloats(rng));

    if (FailureCount != 0) {
        std::printf("\n%d check(s) FAILED\n", FailureCount);
        return 1;
    }
    std::printf("\nall checks passed\n");
    return 0;
}
