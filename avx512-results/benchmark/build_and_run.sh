#!/usr/bin/env bash
# Builds the apply-time benchmark against the C API library and runs it twice:
# once with the AVX-512 evaluator enabled and once with CATBOOST_NO_AVX512=1.
#
#   nix build                       # produces ./result with libcatboostmodel
#   ./avx512-results/benchmark/build_and_run.sh --model path/to/model.cbm
#
# Any extra arguments are passed through to the benchmark (--docs, --repetitions,
# --only, ...). Set CATBOOST_RESULT to point at a different library prefix.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
prefix="${CATBOOST_RESULT:-$repo/result}"
out="${TMPDIR:-/tmp}/catboost_avx512_benchmark"
cxx="${CXX:-clang++}"

if [ ! -f "$prefix/lib/libcatboostmodel.so" ]; then
    echo "no $prefix/lib/libcatboostmodel.so -- run 'nix build' in $repo first" >&2
    exit 1
fi

"$cxx" -std=c++20 -O2 -Wall -Wextra \
    -I"$repo/catboost/libs/model_interface" \
    "$here/benchmark.cpp" \
    -L"$prefix/lib" -lcatboostmodel -Wl,-rpath,"$prefix/lib" \
    -o "$out"

echo "=== AVX-512 evaluator enabled (if the CPU supports it)"
"$out" "$@"

echo
echo "=== baseline (CATBOOST_NO_AVX512=1, Y_NO_AVX512_IN_DOT_PRODUCT=1)"
CATBOOST_NO_AVX512=1 Y_NO_AVX512_IN_DOT_PRODUCT=1 "$out" "$@"
