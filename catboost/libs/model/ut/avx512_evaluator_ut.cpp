#include <catboost/libs/model/model.h>

#include <library/cpp/testing/unittest/registar.h>

#include <util/generic/xrange.h>
#include <util/random/fast.h>

#include <cmath>

// Cross-checks the batched evaluator against the single-document one over
// document counts that straddle every block and register boundary it has.
//
// On a CPU with AVX-512 the batched path is the AVX-512 evaluator working in
// blocks of up to 512 documents, while the single-document path is plain scalar
// code, so the comparison is exactly the AVX-512-versus-baseline check that
// matters. Elsewhere it still guards the block-size plumbing and the SSE path.

using namespace NCB;
using namespace NCB::NModelEvaluation;

namespace {
    // Document counts around the 64-document register, the 128-document baseline
    // block and the 512-document AVX-512 block, plus a few sizes in between.
    const TVector<size_t> DOC_COUNTS = {
        1, 2, 7, 8, 15, 16, 63, 64, 65, 100, 127, 128, 129, 191, 192,
        255, 256, 300, 511, 512, 513, 640, 1000
    };

    TFullModel MakeSyntheticObliviousModel(
        size_t featureCount,
        size_t borderCount,
        size_t treeCount,
        size_t treeDepth,
        size_t approxDimension,
        ui64 seed
    ) {
        TFastRng64 rng(seed);
        TFullModel model;
        TModelTrees* trees = model.ModelTrees.GetMutable();

        TVector<TFloatFeature> floatFeatures;
        for (size_t featureIndex : xrange(featureCount)) {
            TVector<float> borders;
            for (size_t borderIndex : xrange(borderCount)) {
                borders.push_back(float(borderIndex + 1) / float(borderCount + 1));
            }
            floatFeatures.emplace_back(false, int(featureIndex), int(featureIndex), borders, "");
        }
        trees->SetFloatFeatures(floatFeatures);

        TVector<double> leafValues;
        for (size_t treeIndex : xrange(treeCount)) {
            Y_UNUSED(treeIndex);
            TVector<int> splits;
            for (size_t depth : xrange(treeDepth)) {
                Y_UNUSED(depth);
                splits.push_back(int(rng.Uniform(featureCount * borderCount)));
            }
            trees->AddBinTree(splits);
            for (size_t leafIndex : xrange((size_t(1) << treeDepth) * approxDimension)) {
                Y_UNUSED(leafIndex);
                leafValues.push_back(rng.GenRandReal1() * 2.0 - 1.0);
            }
        }
        trees->SetLeafValues(leafValues);
        trees->SetApproxDimension(int(approxDimension));
        model.UpdateDynamicData();
        return model;
    }

    TVector<TVector<float>> MakeFeatures(size_t docCount, size_t featureCount, ui64 seed) {
        TFastRng64 rng(seed);
        TVector<TVector<float>> features(docCount, TVector<float>(featureCount));
        for (auto& document : features) {
            for (auto& value : document) {
                value = float(rng.GenRandReal1());
            }
        }
        return features;
    }

    void CheckModelOnAllDocCounts(const TFullModel& model, ui64 seed) {
        const size_t featureCount = model.GetNumFloatFeatures();
        const size_t approxDimension = model.GetDimensionsCount();

        for (size_t docCount : DOC_COUNTS) {
            const auto documents = MakeFeatures(docCount, featureCount, seed + docCount);

            TVector<TConstArrayRef<float>> documentRefs(docCount);
            for (size_t docId : xrange(docCount)) {
                documentRefs[docId] = documents[docId];
            }

            TVector<TVector<float>> transposed(featureCount, TVector<float>(docCount));
            TVector<TConstArrayRef<float>> transposedRefs(featureCount);
            for (size_t featureId : xrange(featureCount)) {
                for (size_t docId : xrange(docCount)) {
                    transposed[featureId][docId] = documents[docId][featureId];
                }
                transposedRefs[featureId] = transposed[featureId];
            }

            TVector<double> expected(docCount * approxDimension);
            for (size_t docId : xrange(docCount)) {
                model.CalcFlatSingle(
                    documentRefs[docId],
                    TArrayRef<double>(expected.data() + docId * approxDimension, approxDimension));
            }

            TVector<double> transposedResult(docCount * approxDimension);
            model.CalcFlatTransposed(transposedRefs, transposedResult);

            TVector<double> flatResult(docCount * approxDimension);
            model.CalcFlat(documentRefs, flatResult);

            for (size_t index : xrange(expected.size())) {
                UNIT_ASSERT_DOUBLES_EQUAL_C(
                    expected[index], transposedResult[index], 1e-9,
                    "transposed batch differs from single document"
                        << LabeledOutput(docCount, index, expected[index], transposedResult[index]));
                UNIT_ASSERT_DOUBLES_EQUAL_C(
                    expected[index], flatResult[index], 1e-9,
                    "flat batch differs from single document"
                        << LabeledOutput(docCount, index, expected[index], flatResult[index]));
            }
        }
    }

    void CheckLeafIndexesOnAllDocCounts(const TFullModel& model, ui64 seed) {
        const size_t featureCount = model.GetNumFloatFeatures();
        const size_t treeCount = model.GetTreeCount();

        for (size_t docCount : DOC_COUNTS) {
            const auto documents = MakeFeatures(docCount, featureCount, seed + docCount);
            TVector<TConstArrayRef<float>> documentRefs(docCount);
            for (size_t docId : xrange(docCount)) {
                documentRefs[docId] = documents[docId];
            }

            TVector<ui32> batchIndexes(docCount * treeCount);
            model.CalcLeafIndexes(documentRefs, {}, batchIndexes);

            TVector<ui32> singleIndexes(treeCount);
            for (size_t docId : xrange(docCount)) {
                model.CalcLeafIndexesSingle(documentRefs[docId], {}, singleIndexes);
                for (size_t treeId : xrange(treeCount)) {
                    UNIT_ASSERT_VALUES_EQUAL_C(
                        singleIndexes[treeId], batchIndexes[docId * treeCount + treeId],
                        LabeledOutput(docCount, docId, treeId));
                }
            }
        }
    }
}

Y_UNIT_TEST_SUITE(TAvx512Evaluator) {
    // Shallow trees over enough features that a block does not fit a cache line:
    // the shape the AVX-512 index kernel and the leaf gather are built for.
    Y_UNIT_TEST(TestShallowTrees) {
        const auto model = MakeSyntheticObliviousModel(
            /*featureCount*/ 40, /*borderCount*/ 16, /*treeCount*/ 37, /*treeDepth*/ 6,
            /*approxDimension*/ 1, /*seed*/ 20260816);
        CheckModelOnAllDocCounts(model, 1);
    }

    // Tree count not a multiple of four, so both the four-at-a-time loop and the
    // one-tree-at-a-time remainder are exercised, at the maximum depth the
    // one-byte leaf index allows.
    Y_UNIT_TEST(TestMaxDepthForByteIndexes) {
        const auto model = MakeSyntheticObliviousModel(
            /*featureCount*/ 12, /*borderCount*/ 8, /*treeCount*/ 7, /*treeDepth*/ 8,
            /*approxDimension*/ 1, /*seed*/ 20260817);
        CheckModelOnAllDocCounts(model, 2);
    }

    // Deeper than eight: falls back to the 32-bit index path.
    Y_UNIT_TEST(TestDeepTrees) {
        const auto model = MakeSyntheticObliviousModel(
            /*featureCount*/ 16, /*borderCount*/ 4, /*treeCount*/ 5, /*treeDepth*/ 10,
            /*approxDimension*/ 1, /*seed*/ 20260818);
        CheckModelOnAllDocCounts(model, 3);
    }

    Y_UNIT_TEST(TestMultiClass) {
        const auto model = MakeSyntheticObliviousModel(
            /*featureCount*/ 20, /*borderCount*/ 8, /*treeCount*/ 11, /*treeDepth*/ 6,
            /*approxDimension*/ 4, /*seed*/ 20260819);
        CheckModelOnAllDocCounts(model, 4);
    }

    Y_UNIT_TEST(TestLeafIndexes) {
        const auto model = MakeSyntheticObliviousModel(
            /*featureCount*/ 24, /*borderCount*/ 8, /*treeCount*/ 9, /*treeDepth*/ 6,
            /*approxDimension*/ 1, /*seed*/ 20260820);
        CheckLeafIndexesOnAllDocCounts(model, 5);
    }
}
