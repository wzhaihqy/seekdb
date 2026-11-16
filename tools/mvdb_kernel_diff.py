#!/usr/bin/env python3
"""Compare the shipped kernel with the one it replaces, bit for bit.

The selection answers with the rows that tie with the boundary, and the band
decides which rows get their exact float64 distance computed.  A different
band is a different answer, so the two objects are compared on their output
rather than on their speed: every id and every distance has to be equal.

Both objects are loaded at once -- the one the repository ships and the one
given on the command line -- and both are asked the same questions: random
tables of several shapes, clustered columns, columns with many equal values,
columns whose head differs from the rest, and the extremes of the float
range.

Usage:
    python tools/mvdb_kernel_diff.py --old tmp/so-backup/old-aarch64.so
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

TOPK_ARGTYPES = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ctypes.c_int64, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p]


def load(path):
    lib = ctypes.CDLL(path)
    lib.fl_topk.restype = ctypes.c_int64
    lib.fl_topk.argtypes = TOPK_ARGTYPES
    return lib


def call(lib, coarse, norms, matrix, query, keys, rows_of, ids_of, take,
         tolerance, out_ids, out_dists):
    written = lib.fl_topk(
        coarse.ctypes.data, norms.ctypes.data if norms is not None else None,
        matrix.ctypes.data, query.ctypes.data,
        keys.ctypes.data if keys is not None else None,
        rows_of.ctypes.data if rows_of is not None else None,
        ids_of.ctypes.data if ids_of is not None else None,
        coarse.shape[0], query.shape[0], take, 0, tolerance,
        out_ids.ctypes.data, out_dists.ctypes.data)
    if written < 0:
        return None
    return out_ids[:written].copy(), out_dists[:written].copy()


def build_case(rng, rows, dim, take, shape):
    if shape == "uniform":
        matrix = rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)
    elif shape == "clustered":
        centres = rng.uniform(-1.0, 1.0, size=(8, dim)).astype(np.float32)
        pick = rng.integers(0, 8, size=rows)
        matrix = (centres[pick] + rng.normal(0.0, 0.05, size=(rows, dim))).astype(np.float32)
    elif shape == "equal":
        matrix = np.zeros((rows, dim), dtype=np.float32)
        matrix[:, 0] = 1.0
    elif shape == "outlier":
        matrix = rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)
        matrix[0] = 1e30
    elif shape == "tiny":
        matrix = rng.uniform(-1e-30, 1e-30, size=(rows, dim)).astype(np.float32)
    else:
        raise ValueError(shape)
    query = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)
    norms = (matrix.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    coarse = (matrix.astype(np.float64) @ query.astype(np.float64)).astype(np.float32)
    keys = np.arange(rows, dtype=np.int64) + 1
    return matrix, query, norms, coarse, keys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="the object to compare with")
    parser.add_argument("--new", default=os.path.join(
        REPO, "minivectordb", "_fastlane_%s.so" % os.uname().machine))
    parser.add_argument("--cases", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args(argv)

    old = load(args.old)
    new = load(args.new)
    rng = np.random.default_rng(args.seed)

    shapes = ("uniform", "clustered", "equal", "outlier", "tiny")
    sizes = ((1, 8), (2, 8), (10, 16), (137, 32), (1000, 64), (4096, 128),
             (10000, 128), (20000, 16))
    takes = (1, 5, 10, 33)
    tolerances = (1e-3, 0.0, 1e-6)

    mismatches = 0
    compared = 0
    for case in range(args.cases):
        rows, dim = sizes[case % len(sizes)]
        take = takes[case % len(takes)]
        tolerance = tolerances[case % len(tolerances)]
        shape = shapes[case % len(shapes)]
        matrix, query, norms, coarse, keys = build_case(rng, rows, dim, take, shape)
        if case % 3 == 0:
            ids_of = np.arange(rows, dtype=np.int64) * 3 + 7
            rows_of = None
        elif case % 3 == 1:
            ids_of = None
            rows_of = np.arange(rows, dtype=np.int64)
        else:
            ids_of = None
            rows_of = None
        out_ids_a = np.empty(rows, dtype=np.int64)
        out_dists_a = np.empty(rows, dtype=np.float64)
        out_ids_b = np.empty(rows, dtype=np.int64)
        out_dists_b = np.empty(rows, dtype=np.float64)
        a = call(old, coarse, norms, matrix, query, keys, rows_of, ids_of, take,
                 tolerance, out_ids_a, out_dists_a)
        b = call(new, coarse, norms, matrix, query, keys, rows_of, ids_of, take,
                 tolerance, out_ids_b, out_dists_b)
        compared += 1
        if a is None or b is None:
            if a is not b:
                mismatches += 1
                print("case %d: one of them refused the call" % case)
            continue
        if len(a[0]) != len(b[0]):
            mismatches += 1
            print("case %d (%s rows=%d dim=%d take=%d): %d rows vs %d rows"
                  % (case, shape, rows, dim, take, len(a[0]), len(b[0])))
            continue
        if not np.array_equal(a[0], b[0]) or not np.array_equal(a[1], b[1]):
            mismatches += 1
            print("case %d (%s rows=%d dim=%d take=%d): the answers differ"
                  % (case, shape, rows, dim, take))
    print("compared %d cases, %d mismatches" % (compared, mismatches))
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
