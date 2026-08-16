// Apply-time benchmark for the CPU evaluator, driven through the C API so that
// it measures exactly the library `nix build` produces.
//
// It times one batch of documents applied over and over, in the two input
// layouts the evaluator distinguishes:
//   * transposed     -- CalcModelPredictionFlatTransposed, feature-major input;
//   * non-transposed -- CalcModelPredictionFlat, document-major input.
//
// Whether the AVX-512 evaluator is used is decided at run time by the library,
// so an A/B is two runs of this program: one plain, one with
// CATBOOST_NO_AVX512=1 in the environment. The program prints a checksum of the
// predictions, which must be identical across the two runs -- that is the
// correctness half of the comparison.

#include <c_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
    struct TOptions {
        std::string ModelPath;
        size_t DocCount = 1024;
        size_t Repetitions = 100;
        size_t WarmupRepetitions = 10;
        unsigned Seed = 20260816;
        bool Transposed = true;
        bool NonTransposed = true;
    };

    [[noreturn]] void Usage(const char* argv0, int code) {
        std::fprintf(
            code == 0 ? stdout : stderr,
            "usage: %s --model <model.cbm> [options]\n"
            "\n"
            "  --model PATH         model to apply (required)\n"
            "  --docs N             documents per batch (default 1024)\n"
            "  --repetitions N      timed batches (default 100)\n"
            "  --warmup N           untimed batches before timing (default 10)\n"
            "  --seed N             feature generator seed (default 20260816)\n"
            "  --only transposed|flat   run just one layout\n"
            "\n"
            "Set CATBOOST_NO_AVX512=1 to force the baseline evaluator.\n",
            argv0);
        std::exit(code);
    }

    TOptions ParseOptions(int argc, char** argv) {
        TOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> std::string {
                if (i + 1 >= argc) {
                    std::fprintf(stderr, "%s needs a value\n", arg.c_str());
                    Usage(argv[0], 2);
                }
                return argv[++i];
            };
            if (arg == "--model") {
                options.ModelPath = next();
            } else if (arg == "--docs") {
                options.DocCount = std::stoul(next());
            } else if (arg == "--repetitions") {
                options.Repetitions = std::stoul(next());
            } else if (arg == "--warmup") {
                options.WarmupRepetitions = std::stoul(next());
            } else if (arg == "--seed") {
                options.Seed = static_cast<unsigned>(std::stoul(next()));
            } else if (arg == "--only") {
                const std::string value = next();
                if (value == "transposed") {
                    options.NonTransposed = false;
                } else if (value == "flat") {
                    options.Transposed = false;
                } else {
                    std::fprintf(stderr, "--only takes transposed or flat\n");
                    Usage(argv[0], 2);
                }
            } else if (arg == "--help" || arg == "-h") {
                Usage(argv[0], 0);
            } else {
                std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
                Usage(argv[0], 2);
            }
        }
        if (options.ModelPath.empty()) {
            std::fprintf(stderr, "--model is required\n");
            Usage(argv[0], 2);
        }
        return options;
    }

    // Order-independent so that it does not depend on how the batch was split.
    double Checksum(const std::vector<double>& values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum;
    }

    struct TTimings {
        std::vector<double> MillisecondsPerBatch;

        double Min() const {
            return *std::min_element(MillisecondsPerBatch.begin(), MillisecondsPerBatch.end());
        }

        double Mean() const {
            double sum = 0.0;
            for (double value : MillisecondsPerBatch) {
                sum += value;
            }
            return sum / MillisecondsPerBatch.size();
        }

        double Median() const {
            std::vector<double> sorted = MillisecondsPerBatch;
            std::sort(sorted.begin(), sorted.end());
            return sorted[sorted.size() / 2];
        }
    };

    template <class TApply>
    TTimings Measure(const TOptions& options, TApply apply) {
        for (size_t i = 0; i < options.WarmupRepetitions; ++i) {
            apply();
        }
        TTimings timings;
        timings.MillisecondsPerBatch.reserve(options.Repetitions);
        for (size_t i = 0; i < options.Repetitions; ++i) {
            const auto start = std::chrono::steady_clock::now();
            apply();
            const auto finish = std::chrono::steady_clock::now();
            timings.MillisecondsPerBatch.push_back(
                std::chrono::duration<double, std::milli>(finish - start).count());
        }
        return timings;
    }

    void Report(const char* name, const TTimings& timings, size_t docCount, double checksum) {
        const double mean = timings.Mean();
        std::printf(
            "%-16s min %8.3f ms   median %8.3f ms   mean %8.3f ms   %9.0f docs/s   checksum %.9g\n",
            name, timings.Min(), timings.Median(), mean, docCount / (mean / 1000.0), checksum);
    }
}

int main(int argc, char** argv) {
    const TOptions options = ParseOptions(argc, argv);

    ModelCalcerHandle* model = ModelCalcerCreate();
    if (!model) {
        std::fprintf(stderr, "ModelCalcerCreate failed\n");
        return 1;
    }
    if (!LoadFullModelFromFile(model, options.ModelPath.c_str())) {
        std::fprintf(stderr, "failed to load %s: %s\n", options.ModelPath.c_str(), GetErrorString());
        return 1;
    }

    const size_t featureCount = GetFloatFeaturesCount(model);
    const size_t dimension = GetDimensionsCount(model);
    if (GetCatFeaturesCount(model) != 0) {
        std::fprintf(stderr, "models with categorical features are not supported by this benchmark\n");
        return 1;
    }

    std::printf(
        "model %s: %zu float features, %zu trees, approx dimension %zu\n",
        options.ModelPath.c_str(), featureCount, GetTreeCount(model), dimension);
    std::printf(
        "batch %zu documents, %zu repetitions after %zu warmup, CATBOOST_NO_AVX512=%s\n\n",
        options.DocCount, options.Repetitions, options.WarmupRepetitions,
        std::getenv("CATBOOST_NO_AVX512") ? std::getenv("CATBOOST_NO_AVX512") : "(unset)");

    // Feature-major (transposed) storage, and a document-major copy of the very
    // same values so that the two layouts are timed on identical input.
    std::mt19937 rng(options.Seed);
    std::uniform_real_distribution<float> distribution(-3.0f, 3.0f);

    std::vector<float> transposedStorage(featureCount * options.DocCount);
    for (auto& value : transposedStorage) {
        value = distribution(rng);
    }
    std::vector<const float*> transposedRows(featureCount);
    for (size_t featureId = 0; featureId < featureCount; ++featureId) {
        transposedRows[featureId] = transposedStorage.data() + featureId * options.DocCount;
    }

    std::vector<float> flatStorage(featureCount * options.DocCount);
    std::vector<const float*> flatRows(options.DocCount);
    for (size_t docId = 0; docId < options.DocCount; ++docId) {
        flatRows[docId] = flatStorage.data() + docId * featureCount;
        for (size_t featureId = 0; featureId < featureCount; ++featureId) {
            flatStorage[docId * featureCount + featureId] =
                transposedStorage[featureId * options.DocCount + docId];
        }
    }

    std::vector<double> results(options.DocCount * dimension);

    int exitCode = 0;
    double transposedChecksum = 0.0;
    bool haveTransposedChecksum = false;

    if (options.Transposed) {
        const auto timings = Measure(options, [&] {
            if (!CalcModelPredictionFlatTransposed(
                    model, options.DocCount, transposedRows.data(), featureCount,
                    results.data(), results.size()))
            {
                std::fprintf(stderr, "CalcModelPredictionFlatTransposed failed: %s\n", GetErrorString());
                std::exit(1);
            }
        });
        transposedChecksum = Checksum(results);
        haveTransposedChecksum = true;
        Report("transposed", timings, options.DocCount, transposedChecksum);
    }

    if (options.NonTransposed) {
        const auto timings = Measure(options, [&] {
            if (!CalcModelPredictionFlat(
                    model, options.DocCount, flatRows.data(), featureCount,
                    results.data(), results.size()))
            {
                std::fprintf(stderr, "CalcModelPredictionFlat failed: %s\n", GetErrorString());
                std::exit(1);
            }
        });
        const double flatChecksum = Checksum(results);
        Report("non-transposed", timings, options.DocCount, flatChecksum);

        // The two layouts feed the evaluator the same numbers, so any difference
        // beyond floating point noise is a bug in one of the two paths.
        if (haveTransposedChecksum) {
            const double tolerance = 1e-9 * std::max(1.0, std::fabs(transposedChecksum));
            if (std::fabs(flatChecksum - transposedChecksum) > tolerance) {
                std::fprintf(
                    stderr,
                    "\nMISMATCH between layouts: transposed %.17g vs non-transposed %.17g\n",
                    transposedChecksum, flatChecksum);
                exitCode = 1;
            }
        }
    }

    ModelCalcerDelete(model);
    return exitCode;
}
