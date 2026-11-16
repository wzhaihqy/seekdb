#!/usr/bin/env python3
"""Where the time between the C kernel and the Python wrapper goes.

The kernel itself answers a selection in about eight microseconds.  Inside
the engine the same call measures about seventeen.  The difference is not in
the C code -- it is in the way the call is made from Python.

This script measures the layers separately:

  * the raw ctypes call, with every pointer already extracted;
  * the extraction of those pointers, which is what the wrapper does on
    every call;
  * the wrapper as the engine calls it;
  * the two array copies the wrapper makes on the way out.

Usage:
    python mvdb_ctypes_overhead.py [--rounds 2000]
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from minivectordb import fastlane


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args(argv)

    lib = fastlane._library()
    if lib is None:
        print("the kernel is not available on this machine")
        return 1

    rows = args.rows
    dim = args.dim
    rng = np.random.default_rng(20261112)
    matrix = rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)
    query = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)
    norms = (matrix.astype(np.float64) ** 2).sum(axis=1).astype(np.float32)
    coarse = (matrix @ query).astype(np.float32)
    keys = np.arange(1, rows + 1, dtype=np.int64)
    take = 10
    tolerance = 1e-3

    out_ids = np.empty(rows, dtype=np.int64)
    out_dists = np.empty(rows, dtype=np.float64)

    def measure(label, action, rounds=args.rounds):
        for _ in range(50):
            action()
        start = time.perf_counter()
        for _ in range(rounds):
            action()
        elapsed = (time.perf_counter() - start) / rounds * 1e6
        print("  %-34s %8.2f us" % (label, elapsed))
        return elapsed

    print("=== 从 Python 调用内核的各层开销 (rows=%d) ===" % rows)

    # The pointers, extracted once.
    p_coarse = coarse.ctypes.data
    p_norms = norms.ctypes.data
    p_matrix = matrix.ctypes.data
    p_query = query.ctypes.data
    p_keys = keys.ctypes.data
    p_ids = out_ids.ctypes.data
    p_dists = out_dists.ctypes.data

    def raw_call():
        lib.fl_topk(p_coarse, p_norms, p_matrix, p_query, p_keys, None, None,
                    rows, dim, take, 0, tolerance, p_ids, p_dists)

    measure("裸 ctypes 调用（指针已缓存）", raw_call)

    def extract_only():
        coarse.ctypes.data
        norms.ctypes.data
        matrix.ctypes.data
        query.ctypes.data
        keys.ctypes.data
        out_ids.ctypes.data
        out_dists.ctypes.data

    measure("提取 7 个 .ctypes.data", extract_only)

    def flags_only():
        coarse.dtype == np.float32
        query.dtype == np.float32
        matrix.dtype == np.float32
        matrix.flags["C_CONTIGUOUS"]
        coarse.shape[0]
        query.shape[0]

    measure("包装层的形状与类型检查", flags_only)

    measure("包装层 topk() 整体",
             lambda: fastlane.topk(coarse, norms, matrix, query, keys,
                                   None, None, take, tolerance))

    def copy_only():
        out_ids[:10].copy()
        out_dists[:10].copy()

    measure("返回时的两次数组拷贝", copy_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
