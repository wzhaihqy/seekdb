"""How much does a float32 matrix product move the distances?

The scan turns the distances into one matrix product, ``|a-q|^2 = |a|^2 -
2 a.q + |q|^2``.  With float64 operands the product is exact enough; with
float32 operands the product itself carries an error of about 1e-7, and the
cancellation of the formula can add to it.  This script measures the error on
the benchmark shape and checks whether the selection of the closest rows
changes, which is what the candidate tolerance has to cover.

Usage:
    python tools/mvdb_float32_check.py [--rows 10000] [--dim 128] [--k 10]
"""
from __future__ import annotations

import argparse
import math
import random

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def reference_distance(row, query):
    total = 0.0
    for a, b in zip(row, query):
        diff = float(a) - float(b)
        total += diff * diff
    return math.sqrt(total)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args(argv)

    if np is None:
        print("numpy is missing")
        return 1

    rng = random.Random(args.seed)
    matrix = np.array([[rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
                       for _ in range(args.rows)], dtype=np.float32)
    norms = np.einsum("ij,ij->i", matrix.astype(np.float64), matrix.astype(np.float64))
    query_norm_cache = {}

    worst_squared = 0.0
    worst_distance = 0.0
    changed = 0
    for index in range(args.queries):
        query = np.array([rng.uniform(-1.0, 1.0) for _ in range(args.dim)], dtype=np.float32)
        query64 = query.astype(np.float64)
        query_norm = float(np.dot(query64, query64))

        exact = norms - 2.0 * (matrix.astype(np.float64) @ query64) + query_norm
        np.maximum(exact, 0.0, out=exact)

        product = matrix @ query
        approximate = norms - 2.0 * product.astype(np.float64) + query_norm
        np.maximum(approximate, 0.0, out=approximate)

        difference = np.abs(exact - approximate)
        worst_squared = max(worst_squared, float(difference.max()))
        worst_distance = max(worst_distance,
                             float(np.sqrt(np.maximum(exact, 0.0)).max()
                                   - np.sqrt(np.maximum(approximate, 0.0)).min()))

        exact_top = set(np.argsort(exact)[:args.k].tolist())
        approximate_top = set(np.argsort(approximate)[:args.k].tolist())
        if exact_top != approximate_top:
            changed += 1

    # the boundary gap: how far apart are the k-th and the (k+1)-th distance
    query = np.array([rng.uniform(-1.0, 1.0) for _ in range(args.dim)], dtype=np.float32)
    query64 = query.astype(np.float64)
    exact = norms - 2.0 * (matrix.astype(np.float64) @ query64) + float(np.dot(query64, query64))
    np.maximum(exact, 0.0, out=exact)
    ordered = np.sort(exact)
    gap = float(ordered[args.k] - ordered[args.k - 1])

    print("rows %d dim %d queries %d" % (args.rows, args.dim, args.queries))
    print("worst squared distance error : %.3e" % worst_squared)
    print("worst distance error         : %.3e" % (worst_distance / max(1e-12, 1.0)))
    print("top %d sets that differ      : %d" % (args.k, changed))
    print("gap between the k-th and the (k+1)-th squared distance: %.3e" % gap)
    print("a tolerance of 1e-5 covers the error by a factor of %.1f"
          % (1e-5 / max(1e-12, worst_squared)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
