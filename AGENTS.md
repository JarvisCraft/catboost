# AGENTS.md

CatBoost — Yandex's gradient boosting library. Arcadia-style C++20 core (from the Yandex monorepo) with Python/R/Java/Spark/Node bindings. This OSS tree ships without the `ya.make` files; the checked-in, YaTool-generated CMake files are the operative build config.

## Build

- Native build wrapper (drives Conan + CMake + Ninja): `python build/build_native.py --build-root-dir=<dir> --targets catboost`. Use `--dry-run --verbose` to inspect commands before running. Requires Conan 2; deps come from `conanfile.py` via `cmake/conan_provider.cmake`.
- CUDA is OFF by default; enable with `--have-cuda` (or `-DHAVE_CUDA=yes`) plus `--cuda-root-dir=<path>`.
- CI builds via `ci/build_all.py` (inside a manylinux2014 docker, cross-compiles for aarch64). For local work prefer `build_native.py`.
- In-source builds are forbidden (enforced by the root `CMakeLists.txt`).
- Key targets (see `Targets` in `build/build_native.py`):
  - `catboost` — CLI app → `<build>/catboost/app/catboost`
  - `catboostmodel`, `catboostmodel_static` — C++ model-interface libs
  - `_catboost`, `_hnsw` — Python extension libs
  - `catboostr` — R-package lib
  - test tools: `limited_precision_dsv_diff`, `limited_precision_json_diff`, `model_comparator`
  - build tools (only needed for cross-compilation): `archiver`, `protoc`, `flatc`, `enum_parser`, `rescompiler`, `triecompiler`

### CMake files are generated and per-platform

- Build config is a stack of generated `CMakeLists.<platform>.txt` files (e.g. `CMakeLists.linux-x86_64.txt`, `CMakeLists.linux-x86_64-cuda.txt`, ...) present at repo root and in every buildable directory; the root `CMakeLists.txt` picks the file by OS/arch/`HAVE_CUDA`. They are regenerated from internal YaTool `ya.make` files by the maintainers.
- Gotcha: each platform file explicitly enumerates `target_sources(...)`. Adding a `.cpp` to a target means adding it to every platform file you want to keep building (they differ only in CUDA bits) — don't add it to just one.
- Custom CMake macros live in `cmake/*.cmake`: `add_recursive_library`, `add_yunittest`, `target_allocator`, `generate_enum_serilization`, etc.

## Test

- C++ unit tests: `ut/` subdirs; target name is the dashed lib path (`catboost/libs/model/ut` → `model_ut`), registered via `add_yunittest`. Tests use the ytest framework, not gtest — write `Y_UNIT_TEST(...)` (from `library/cpp/testing/unittest`). Build the target, then run the executable or `ctest -R model_ut`.
- CLI functional suite (the big one), from `catboost/pytest`:
  - needs the `catboost` CLI binary, the `limited_precision_dsv_diff` tool, and an installed `catboost` python package (used for reading column descriptions)
  - env vars: `CMAKE_BINARY_DIR` (CMake build root), `TEST_OUTPUT_DIR`, `PORT_SYNC_PATH`, `HAVE_CUDA=0` (or `1` for GPU). `CMAKE_SOURCE_DIR` is not needed — the repo root is auto-detected from git.
  - run: `python -m pytest -n auto`
  - GPU-only tests are in `cuda_tests/` and self-skip when `HAVE_CUDA=0`.
- Python-package tests: install the built wheel first, then `python -m pytest -n auto` in `catboost/python-package/ut/medium` and `ut/large` (same env vars minus `HAVE_CUDA`; also needs `limited_precision_json_diff` and `model_comparator`).
- JVM: `mvn test` in `catboost/jvm-packages/catboost4j-prediction` (GPU via `-DtestOnGPU=1`). Spark: run `catboost/spark/catboost4j-spark/generate_projects/generate.py` first. R: `R CMD check .` or `devtools::test()` in `catboost/R-package`.

## Structure

- `catboost/` — the product: `app` (CLI), `libs` (core libs), `private/libs` (implementation internals — algo, options, quantization, ...), `cuda`, `python-package`, `R-package`, `jvm-packages`, `spark`, `tools`, `pytest` (functional tests), `docs`.
- `library/` — shared Arcadia libs: `library/cpp/...` → CMake target `library-cpp-<name>`; `library/python/...` (ytest harness, port_manager).
- `util/` — Arcadia `util` (TString, yvector, ...), `contrib/` — vendored third-party. Treat both as third-party; avoid editing casually.
- `build/` — build infra (`build_native.py`, `scripts/`), `cmake/` — CMake macros + Conan profiles, `ci/` — CI scripts + toolchains, `tools/` — linters + build tools.

## Conventions

- C++ must follow `CPP_STYLE_GUIDE.md` and the (Russian) `catboost_command_style_guide_extension.md`: `T`-prefixed class names, lowercase type suffixes (`ui64`, `i32`, `TString`), verbs-first function names (`GetFeatureCount`, `CountFeatures`), lowercase abbreviations (`TDsvWriter`), anonymous-namespace for classes not declared in a `.h`, one-letter vars only for `for` loop indices, file names with underscores. There is no `.clang-format`; style is enforced by the `cpp_styleguide` build tool.
- Python follows PEP8; checkers are `tools/black_linter`, `tools/flake8_linter`, `tools/ruff_linter`.
- Do not bump package versions in PRs — the maintainers version internally (see docs "Versioning conventions").
- Commits must carry the trailer `Assisted-by: OpenCode <dev@opencode.ai>` when generated with AI assistance.
