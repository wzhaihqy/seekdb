#!/usr/bin/env python3
"""Time the pieces of a query inside the engine, on the real path.

The pieces are measured by wrapping them, not by reading a model of them:
``numpy.dot`` and ``fastlane.topk`` are the two calls that cost something,
and the answer the engine returns is checked against the one the unwrapped
engine returns, so a wrapper that changes the answer cannot pass unnoticed.

Usage:
    PYTHONPATH=seekdb/repo python tools/mvdb_piece_profile.py [--rows 10000]
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, WS)

import numpy as np                                      # noqa: E402

from minivectordb import fastlane                       # noqa: E402
from minivectordb.engine import Engine                  # noqa: E402
from minivectordb.sqlparser import parse                # noqa: E402


def literal(values):
    return "[" + ",".join("%.6f" % v for v in values) + "]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    data_dir = os.path.join(WS, "..", "tmp", "piece-profile")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)

    engine = Engine(data_dir)
    engine.open()
    engine.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                   "content VARCHAR(64), embedding VECTOR(%d));" % args.dim)
    rows = []
    for index in range(args.rows):
        rows.append((index + 1, "cat-%d" % (index % args.categories),
                     "payload-%d" % index,
                     [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]))
    for offset in range(0, len(rows), 100):
        chunk = rows[offset:offset + 100]
        values = ",".join("(%d,'%s','%s','%s')" % (pk, cat, content, literal(vec))
                          for pk, cat, content, vec in chunk)
        engine.execute("INSERT INTO docs VALUES %s;" % values)

    statements = []
    for index in range(args.queries):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
        if index % 2 == 0:
            statements.append(
                "SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                "WHERE category = 'cat-%d' ORDER BY distance LIMIT 10;"
                % (literal(vector), index % args.categories))
        else:
            statements.append(
                "SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                "ORDER BY distance LIMIT 10;" % literal(vector))

    # The unwrapped answers, to check against later.
    reference = [engine.execute(sql, "exact") for sql in statements[:20]]

    totals = {"dot": 0.0, "topk": 0.0, "parse_text": 0.0, "execute": 0.0,
              "ctypes_call": 0.0}

    original_dot = np.dot

    def timed_dot(*a, **kw):
        start = time.perf_counter()
        out = original_dot(*a, **kw)
        totals["dot"] += time.perf_counter() - start
        return out

    original_topk = fastlane.topk

    def timed_topk(*a, **kw):
        start = time.perf_counter()
        out = original_topk(*a, **kw)
        totals["topk"] += time.perf_counter() - start
        return out

    original_parse_text = fastlane.parse_text

    def timed_parse_text(*a, **kw):
        start = time.perf_counter()
        out = original_parse_text(*a, **kw)
        totals["parse_text"] += time.perf_counter() - start
        return out

    lib = fastlane._library()
    original_call = lib.fl_topk

    def timed_call(*a, **kw):
        start = time.perf_counter()
        out = original_call(*a, **kw)
        totals["ctypes_call"] += time.perf_counter() - start
        return out

    lib.fl_topk = timed_call

    np.dot = timed_dot
    fastlane.topk = timed_topk
    fastlane.parse_text = timed_parse_text

    start = time.perf_counter()
    for sql in statements:
        engine.execute(sql, "exact")
    totals["execute"] = time.perf_counter() - start

    np.dot = original_dot
    fastlane.topk = original_topk
    fastlane.parse_text = original_parse_text
    lib.fl_topk = original_call

    count = len(statements)
    print("queries: %d" % count)
    for label in ("execute", "dot", "topk", "ctypes_call", "parse_text"):
        print("  %-12s %8.2f us/query" % (label, totals[label] / count * 1e6))
    print("  %-12s %8.2f us/query" % ("python around the call",
                                       (totals["topk"] - totals["ctypes_call"])
                                       / count * 1e6))
    print("  %-12s %8.2f us/query" % ("unaccounted",
                                       (totals["execute"] - totals["dot"]
                                        - totals["topk"] - totals["parse_text"])
                                       / count * 1e6))

    # The answers must not have changed.
    again = [engine.execute(sql, "exact") for sql in statements[:20]]
    same = all(a == b for a, b in zip(reference, again))
    print("answers unchanged:", same)
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
