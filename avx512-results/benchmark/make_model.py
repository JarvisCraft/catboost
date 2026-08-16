#!/usr/bin/env python3
"""Trains a synthetic model shaped like the one the reference paper measures.

The paper benchmarks `epsilon8k_64`: binary classification, 2000 float features,
64 borders per feature, 8000 trees of depth 6. Training that here would take
much longer than it is worth, so the shape is a parameter and the default is the
same model with fewer trees. The data is random -- the evaluator is branch-free,
so the numbers only affect which leaves are reached, not how long it takes.

Needs a released catboost python package, which the flake provides in a separate
shell:

    nix develop .#benchmark -c python3 avx512-results/benchmark/make_model.py \\
        --output /tmp/bench.cbm
"""

import argparse

import numpy as np
from catboost import CatBoostClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="where to write the .cbm model")
    parser.add_argument("--features", type=int, default=2000)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--trees", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--borders", type=int, default=64)
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    features = rng.standard_normal((args.rows, args.features)).astype(np.float32)
    # A target that actually depends on the features, so the trees do not all
    # collapse to the same split.
    weights = rng.standard_normal(args.features).astype(np.float32)
    labels = (features @ weights > 0).astype(np.int32)

    model = CatBoostClassifier(
        iterations=args.trees,
        depth=args.depth,
        border_count=args.borders,
        learning_rate=0.1,
        thread_count=args.threads,
        random_seed=args.seed,
        boosting_type="Plain",
        bootstrap_type="No",
        verbose=max(1, args.trees // 10),
    )
    model.fit(features, labels)
    model.save_model(args.output)
    print(f"wrote {args.output}: {args.features} features, {model.tree_count_} trees, depth {args.depth}")


if __name__ == "__main__":
    main()
