#!/usr/bin/env bash
# Builds and runs the AVX-512 kernel check.
#
#   ./build_and_run.sh          # native AVX-512 if the host has it, SIMDe otherwise
#   ./build_and_run.sh simde    # force the SIMDe build
#   ./build_and_run.sh native   # force the native build
#
# The SIMDe build needs the simde headers; with nix they come from
# `nix build nixpkgs#simde`, otherwise point SIMDE_INCLUDE at them.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${TMPDIR:-/tmp}/catboost_avx512_kernel_check"
cxx="${CXX:-clang++}"

mode="${1:-auto}"
if [ "$mode" = "auto" ]; then
    if grep -qw avx512f /proc/cpuinfo 2>/dev/null; then
        mode=native
    else
        mode=simde
    fi
fi

# -Wno-psabi: the SIMDe build has no -mavx*, so passing vector types by value
# trips the "ABI changes" note on gcc. Irrelevant for a single-TU check.
# stub_include stands in for the arcadia util headers, which do not compile
# outside the arcadia libc++ setup.
common=(-std=c++20 -O2 -Wall -Wextra -Werror -Wno-psabi
        -I"$here" -I"$here/stub_include" -I"$here/../..")

checks=(check_kernels check_dot_product)

case "$mode" in
    native)
        for check in "${checks[@]}"; do
            "$cxx" "${common[@]}" \
                -mavx512f -mavx512bw -mavx512dq -mavx512vl \
                "$here/$check.cpp" -o "$out.$check"
        done
        ;;
    simde)
        if [ -z "${SIMDE_INCLUDE:-}" ]; then
            simde_store="$(nix build --no-link --print-out-paths nixpkgs#simde)"
            SIMDE_INCLUDE="$simde_store/include"
        fi
        for check in "${checks[@]}"; do
            "$cxx" "${common[@]}" \
                -DCATBOOST_AVX512_KERNELS_WITH_SIMDE=1 \
                -I"$SIMDE_INCLUDE" \
                "$here/$check.cpp" -o "$out.$check"
        done
        ;;
    *)
        echo "unknown mode: $mode" >&2
        exit 2
        ;;
esac

status=0
for check in "${checks[@]}"; do
    echo "--- $check"
    "$out.$check" || status=1
    echo
done
exit $status
