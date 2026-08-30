# AVX-512 CatBoost implementation

# Problem

Currently, CatBoost implements vectorized computations using SSE3 or ar most AVX2 intrinsics despite availability of AVX-512 on modern server CPUs.

# Goal

Implement optimized AVX-512 functions and measure their performance compared to current (baseline) SSE implementation.

At first, you may only work on transposed path, as it is something that I personally need. Non-transposed *may* be left for later.

The first place to look at is probably `catboost/libs/model/cpu/evaluator_impl.cpp` with its `CalcIndexesSse` function and `library/cpp/dot_product/dot_product_*`.

# Prior art

There is a student paper on using AVX-512 for CatBoost. The paper is available as `./avx512-results/paper.pdf` (written on Russian). It also contains useful information on benchmarking. For your own good it may be worth reading it and summarizing the essential details.

This implementation is available at `../catboost-cpu-evaluation-optimization` which is an old checkout of the current repository plus the single latest commit implementing AVX-512 optimization. It may be used as reference, but the code there may not be up-to-date and there may be bugs in it, so it may be worth reimplementing the optimization only looking at this project as reference.

# Recommendations

- There is a Nix flake configuring basic environment for this project. If some tool is needed, prefer adding it to `flake.nix`.
- Prefer using subagents for atomic subtasks.
- Record important observations and benchmark results in markdown docs in `./avx512-results/`.
- Run benchmarks in separate processes (using `zellij`) so that the dialog is not blocked while they are running.
- Use `nix build` to build the project.
