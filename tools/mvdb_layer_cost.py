#!/usr/bin/env python3
"""Where the 284 us of one TopK query goes, layer by layer.

The judge charges the whole query to the score, so this tool times the
individual steps of one scan on the workload's own shape (10,000 rows x 128
dimensions, K = 10) and prints them side by side:

    np.dot over the column          the coarse ranking pass
    C kernel fl_topk                the same pass in the shipped kernel
    argpartition + lexsort          the selection
    the float64 recomputation       the exact distances of the survivors
    the filter gather               what the half-filtered queries pay

Run with the same interpreter the judge image uses, otherwise the numbers
describe a different machine.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from minivectordb import fastlane          # noqa: E402


def median_us(fn, rounds=24, warmup=5):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)

    if args.threads:
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[name] = str(args.threads)

    rng = np.random.default_rng(20261112)
    matrix = rng.uniform(-1.0, 1.0, size=(args.rows, args.dim)).astype(np.float32)
    query = rng.uniform(-1.0, 1.0, size=args.dim).astype(np.float32)
    norms = np.einsum("ij,ij->i", matrix, matrix).astype(np.float32)
    pks = np.arange(1, args.rows + 1, dtype=np.int64)

    print("=== 一次 TopK 查询的成本分解 (rows=%d dim=%d k=%d) ===" % (args.rows, args.dim, args.k))
    print("numpy", np.__version__, "| BLAS threads:",
          os.environ.get("OPENBLAS_NUM_THREADS", "(default)"))
    print("fastlane available:", fastlane.available())
    print()

    # 1) coarse ranking pass in numpy: |a|^2 - 2 a.q
    out = np.empty(args.rows, dtype=np.float32)

    def numpy_dot():
        np.dot(matrix, query, out=out)
        np.multiply(out, -2.0, out=out)

    t_dot = median_us(numpy_dot)

    # 2) the same pass in the C kernel
    lib = fastlane._library()
    t_kernel = None
    if lib is not None:
        ids = np.empty(args.rows, dtype=np.int64)
        dists = np.empty(args.rows, dtype=np.float64)

        def kernel_call():
            lib.fl_topk(
                ctypes.c_void_p(out.ctypes.data), ctypes.c_void_p(norms.ctypes.data),
                ctypes.c_void_p(matrix.ctypes.data), ctypes.c_void_p(query.ctypes.data),
                ctypes.c_void_p(pks.ctypes.data), None, None,
                args.rows, args.dim, args.k, 0, ctypes.c_double(1e-3),
                ctypes.c_void_p(ids.ctypes.data), ctypes.c_void_p(dists.ctypes.data))

        t_kernel = median_us(kernel_call)

    # 3) selection on the numpy distances
    squared = np.empty(args.rows, dtype=np.float32)
    np.dot(matrix, query, out=squared)

    def select():
        partition = np.argpartition(squared, args.k - 1)[:args.k]
        boundary = squared[partition].max()
        candidates = np.flatnonzero(squared <= boundary + max(1e-6, abs(float(boundary)) * 1e-9))
        order = np.lexsort((pks[candidates], squared[candidates]))
        return candidates[order][:args.k]

    t_select = median_us(select)

    # 4) the float64 recomputation of the survivors
    slots = select()

    def recompute():
        wide = matrix[slots].astype(np.float64)
        wide -= query.astype(np.float64)
        wide *= wide
        return np.sqrt(wide.cumsum(axis=1)[:, -1])

    t_recompute = median_us(recompute)

    # 5) the filter gather: half the queries narrow the table first
    def gather():
        return matrix[::8]

    t_gather = median_us(gather)

    # 6) the reference einsum path (what fastpath.l2_distances does)
    def einsum_path():
        diff = matrix.astype(np.float64, copy=False)
        diff -= query.astype(np.float64)
        sq = np.einsum("ij,ij->i", diff, diff)
        return np.sqrt(sq)

    t_einsum = median_us(einsum_path, rounds=8)

    print("  np.dot 全列扫描 + 系数      : %8.1f us" % t_dot)
    if t_kernel is not None:
        print("  C 内核 fl_topk（含选择）    : %8.1f us" % t_kernel)
    print("  argpartition+lexsort 选择    : %8.1f us" % t_select)
    print("  float64 精确重算(K 行)      : %8.1f us" % t_recompute)
    print("  过滤 gather (1/8 行)        : %8.1f us" % t_gather)
    print("  参考 einsum 全路径          : %8.1f us" % t_einsum)
    print()
    print("  合计(无过滤, numpy 路径)     : %8.1f us" % (t_dot + t_select + t_recompute))
    if t_kernel is not None:
        print("  合计(无过滤, C 内核路径)    : %8.1f us" % (t_kernel + t_recompute))
    print()
    print("  数据规模: %.2f MiB (float32 %dx%d)" % (matrix.nbytes / 1048576.0, args.rows, args.dim))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
