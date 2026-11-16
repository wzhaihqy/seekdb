#!/usr/bin/env python3
"""What the ctypes call into the kernel costs, piece by piece.

The kernel is reached through ctypes, and the call is made once per query.
This script times the parts of that call: the conversion of the arguments,
the call itself, and the copies the answer is wrapped in.  The cost of the
conversion is measured with a call that does no work at all -- the kernel
returns at once when the candidate list is empty -- so what is left is the
price of the call itself.

Usage:
    PYTHONPATH=seekdb/repo python tools/mvdb_call_cost.py
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, WS)

from minivectordb import fastlane                      # noqa: E402


def best_of(fn, rounds=9, inner=50):
    out = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        out.append((time.perf_counter() - start) / inner)
    return min(out) * 1e6


def main():
    lib = fastlane._library()
    rows = 10000
    dim = 128
    rng = np.random.default_rng(1)
    matrix = np.ascontiguousarray(rng.random((rows, dim), dtype=np.float32))
    query = rng.random(dim, dtype=np.float32)
    coarse = np.empty(rows, dtype=np.float32)
    np.dot(matrix, query, out=coarse)
    norms = np.ascontiguousarray(rng.random(rows, dtype=np.float32))
    keys = np.arange(rows, dtype=np.int64)
    out_ids = np.empty(rows, dtype=np.int64)
    out_dists = np.empty(rows, dtype=np.float64)

    raw = [coarse.ctypes.data, norms.ctypes.data, matrix.ctypes.data,
           query.ctypes.data, keys.ctypes.data, out_ids.ctypes.data,
           out_dists.ctypes.data]
    void = [ctypes.c_void_p(value) for value in raw]

    print("=== the call, as the engine makes it ===")
    print("  %-42s %8.2f us" % ("fastlane.topk()", best_of(
        lambda: fastlane.topk(coarse, norms, matrix, query, keys, None, None, 10, 1e-3))))

    print("\n=== the same call, written out ===")
    print("  %-42s %8.2f us" % ("fl_topk, plain ints", best_of(
        lambda: lib.fl_topk(raw[0], raw[1], raw[2], raw[3], raw[4], None, None,
                            rows, dim, 10, 0, 1e-3, raw[5], raw[6]))))
    print("  %-42s %8.2f us" % ("fl_topk, c_void_p objects", best_of(
        lambda: lib.fl_topk(void[0], void[1], void[2], void[3], void[4], None, None,
                            rows, dim, 10, 0, 1e-3, void[5], void[6]))))

    print("\n=== the price of the call itself (no work inside) ===")
    print("  %-42s %8.2f us" % ("14 arguments, plain ints", best_of(
        lambda: lib.fl_topk(raw[0], raw[1], raw[2], raw[3], raw[4], None, None,
                            0, dim, 10, 0, 1e-3, raw[5], raw[6]))))
    print("  %-42s %8.2f us" % ("14 arguments, c_void_p objects", best_of(
        lambda: lib.fl_topk(void[0], void[1], void[2], void[3], void[4], None, None,
                            0, dim, 10, 0, 1e-3, void[5], void[6]))))
    print("  %-42s %8.2f us" % ("one argument (parse_vector)", best_of(
        lambda: lib.fl_parse_vector(b"[1]", 1, ctypes.cast(void[0],
                                                            ctypes.POINTER(ctypes.c_float)),
                                    ctypes.byref(ctypes.c_int(1))))))

    print("\n=== the pieces ===")
    print("  %-42s %8.2f us" % ("seven .ctypes.data", best_of(
        lambda: (coarse.ctypes.data, norms.ctypes.data, matrix.ctypes.data,
                  query.ctypes.data, keys.ctypes.data, out_ids.ctypes.data,
                  out_dists.ctypes.data))))
    print("  %-42s %8.2f us" % ("matrix.flags['C_CONTIGUOUS']", best_of(
        lambda: matrix.flags["C_CONTIGUOUS"])))
    print("  %-42s %8.2f us" % ("two dtype comparisons", best_of(
        lambda: coarse.dtype != np.float32 or query.dtype != np.float32)))
    print("  %-42s %8.2f us" % ("the two copies of the answer", best_of(
        lambda: (out_ids[:10].copy(), out_dists[:10].copy()))))
    print("  %-42s %8.2f us" % ("fastlane._library()", best_of(
        lambda: fastlane._library())))
    print("  %-42s %8.2f us" % ("fastlane.available()", best_of(
        lambda: fastlane.available())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
