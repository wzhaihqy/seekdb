#!/usr/bin/env python3
"""Is the vector column slower to read through a mapping than in RAM?

The engine reads the vector column through a memory map, so every product
touches pages the operating system keeps for it.  The pages are cached, but
a mapped page is still a page the processor has to translate, and the
translation buffer is small.  The alternative is to read the column into
ordinary memory once per write generation and keep it there.

This script measures the three things that decide the question:

  * the product over a mapped column, the way the engine does it;
  * the product over the same column held in ordinary memory;
  * the product with one and with two BLAS workers.

The mapped and the resident copies hold the same bytes, so the answers are
compared as well -- a faster path that returns different numbers would be
worthless.

Usage:
    python mvdb_column_read.py [--rows 10000] [--dim 128]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from minivectordb import _calibrate


def product_time(matrix, query, rounds=60):
    """The median time of one product over the column."""
    out = np.empty(matrix.shape[0], dtype=np.float32)
    for _ in range(5):
        np.dot(matrix, query, out=out)
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        np.dot(matrix, query, out=out)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2] * 1e6


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args(argv)

    rows = args.rows
    dim = args.dim
    root = tempfile.mkdtemp(prefix="mvdb-col-",
                            dir=os.path.join(os.path.dirname(REPO), "tmp"))
    path = os.path.join(root, "c0.dat")

    rng = np.random.default_rng(20261112)
    resident = rng.uniform(-1.0, 1.0, size=(rows, dim)).astype(np.float32)
    resident.tofile(path)

    mapped = np.memmap(path, dtype="<f4", mode="r", shape=(rows, dim))
    asarray_view = np.asarray(mapped)
    query = rng.uniform(-1.0, 1.0, size=dim).astype(np.float32)

    print("=== 向量列的读取方式 (rows=%d dim=%d) ===" % (rows, dim))
    print("  列大小: %.2f MiB" % (rows * dim * 4 / 1024.0 / 1024.0))

    # The calibration decides how many BLAS workers the process uses; the
    # engine asks for it at start-up, so this script asks too.
    verdict = _calibrate.calibrate(rows=rows, dims=dim)
    if verdict:
        threads, two_us, one_us = verdict
        print("  BLAS 校准: %d worker (two=%.1fus one=%.1fus)" % (threads, two_us, one_us))

    mapped_us = product_time(asarray_view, query)
    resident_us = product_time(resident, query)
    print("  memmap 映射的列      : %8.2f us  (%.1f GB/s)"
          % (mapped_us, rows * dim * 4 / mapped_us / 1e3))
    print("  常驻内存的列          : %8.2f us  (%.1f GB/s)"
          % (resident_us, rows * dim * 4 / resident_us / 1e3))
    print("  常驻/映射             : %8.2fx" % (mapped_us / resident_us))

    # The two must produce the same numbers.
    left = np.dot(asarray_view, query)
    right = np.dot(resident, query)
    print("  两列的乘积逐位相同     : %s" % bool(np.array_equal(left, right)))

    # A resident copy is a copy: where does the time go?
    copy_start = time.perf_counter()
    for _ in range(5):
        again = np.fromfile(path, dtype="<f4").reshape(rows, dim)
    copy_us = (time.perf_counter() - copy_start) / 5 * 1e6
    print("  从文件读回常驻        : %8.2f us  (%.1f GB/s)"
          % (copy_us, rows * dim * 4 / copy_us / 1e3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
