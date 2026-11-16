#!/usr/bin/env python3
"""Where the milliseconds of one query actually go.

The bench averages two query shapes: half of them scan the whole table
(ten thousand rows) and half of them carry an equality filter that keeps an
eighth of it (twelve hundred and fifty rows).  The two shapes cost very
different amounts, so an average hides where the time is.  This script
measures the parts of both shapes separately, on the same data, in the same
process, so the numbers can be compared with each other.

Parts measured, in the order the engine runs them:

  dot          numpy.dot(rows, query) -- the float32 product
  squared      the two in place passes that turn the product into |a-q|^2
  kernel call  the ctypes call into the C object, pointers already extracted
  wrapper      fastlane.topk() as the engine calls it, pointers extracted
               inside -- the difference against "kernel call" is what the
               Python layer costs
  engine       engine.execute(sql) as a whole

Usage:
    PYTHONPATH=seekdb/repo python tools/mvdb_path_cost.py [--rows 10000]
"""
from __future__ import annotations

import argparse
import ctypes
import os
import random
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, WS)

from minivectordb import fastlane                      # noqa: E402
from minivectordb.engine import Engine                 # noqa: E402


def best_of(fn, rounds, inner=20):
    """The smallest average of ``inner`` calls, over ``rounds`` rounds.

    The smallest round is the one that met the least interference, so it is
    the closest thing this machine can do.
    """

    out = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        out.append((time.perf_counter() - start) / inner)
    return min(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    data_dir = os.path.join(WS, "..", "tmp", "path-cost")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)

    engine = Engine(data_dir)
    engine.open()
    engine.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                   "content VARCHAR(64), embedding VECTOR(%d));" % args.dim)
    rows = []
    for index in range(args.rows):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
        rows.append((index + 1, "cat-%d" % (index % args.categories),
                     "payload-%d" % index, vector))
    for offset in range(0, len(rows), 100):
        chunk = rows[offset:offset + 100]
        values = ",".join("(%d,'%s','%s','%s')" % (pk, cat, content,
                                                    "[" + ",".join("%.6f" % v for v in vec) + "]")
                          for pk, cat, content, vec in chunk)
        engine.execute("INSERT INTO docs VALUES %s;" % values)
    engine.execute("CREATE VECTOR INDEX docs_vec ON docs(embedding);")

    store = engine.tables["docs"].store

    print("fastlane loaded:", fastlane.available())

    # The two shapes: the whole table, and one category of it.
    whole = engine._view(store, 3)
    matrix, norms, pks, all_live = whole
    ids = np.flatnonzero(np.arange(args.rows) % args.categories == 0)
    block = np.ascontiguousarray(matrix[ids])
    block_norms = norms[ids]
    block_pks = pks[ids]

    query = np.asarray([rng.uniform(-1.0, 1.0) for _ in range(args.dim)],
                       dtype=np.float32)

    def measure(label, fn):
        us = best_of(fn, args.rounds) * 1e6
        print("  %-34s %8.2f us" % (label, us))
        return us

    lib = fastlane._library()
    for name, mtx, nrm, ks, cnt in (("whole table", matrix, norms, pks, args.rows),
                                    ("filtered 1/8", block, block_norms, block_pks,
                                     int(ids.size))):
        print("\n=== %s : %d rows ===" % (name, cnt))
        out = np.empty(cnt, dtype=np.float32)
        out2 = np.empty(cnt, dtype=np.float32)
        measure("numpy.dot(rows, q, out=)", lambda: np.dot(mtx, query, out=out))
        np.dot(mtx, query, out=out)

        def squared():
            np.dot(mtx, query, out=out)
            out[:] *= -2.0
            out[:] += nrm
        measure("dot + the two in place passes", squared)

        # The ctypes call with every pointer already extracted.
        c_mat = mtx.ctypes.data
        c_nrm = nrm.ctypes.data
        c_q = query.ctypes.data
        c_keys = ks.ctypes.data
        c_coarse = out.ctypes.data
        out_slots = np.empty(cnt, dtype=np.intp)
        out_dists = np.empty(cnt, dtype=np.float64)
        c_slots = out_slots.ctypes.data
        c_dists = out_dists.ctypes.data
        np.dot(mtx, query, out=out)

        def raw_call():
            lib.fl_topk(c_coarse, c_nrm, c_mat, c_q, c_keys, None, None,
                        cnt, args.dim, 10, 0, 1e-3, c_slots, c_dists)
        measure("fl_topk, pointers ready", raw_call)

        def wrapped():
            fastlane.topk(out, nrm, mtx, query, ks, None, None, 10, 1e-3)
        measure("fastlane.topk() as the engine calls it", wrapped)

        def pointer_only():
            mtx.ctypes.data, nrm.ctypes.data, query.ctypes.data, ks.ctypes.data
        measure("four .ctypes.data lookups", pointer_only)

    # The engine, end to end, for the two shapes.
    def literal(values):
        return "[" + ",".join("%.6f" % v for v in values) + "]"

    qv = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
    sql_all = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
               "ORDER BY distance LIMIT 10;" % literal(qv))
    sql_filtered = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                    "WHERE category = 'cat-0' ORDER BY distance LIMIT 10;" % literal(qv))
    print("\n=== engine.execute() ===")
    measure("engine.execute, whole table", lambda: engine.execute(sql_all, "exact"))
    measure("engine.execute, filtered 1/8", lambda: engine.execute(sql_filtered, "exact"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
