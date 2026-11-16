#!/usr/bin/env python3
"""Measure the Python glue around the vector scan, inside the engine.

``mvdb_engine_profile.py`` walks the same steps by hand: it parses the
literal with the reference converter, recomputes the exact distances and
rebuilds the answer rows itself.  That is a *model* of the query, and a
model is slower than the thing it models -- the literal the engine really
uses is the C parser, the engine really uses the kernel's exact distances,
and the engine really takes the key-only shortcut.

This script instead calls ``engine.execute(sql)`` -- the real entry point --
and wraps the pieces it is made of.  What is left over after subtracting
the wrapped pieces is the glue: the dictionary lookups, the list building
and the attribute accesses that no profiler would call interesting and that
together cost more than the selection.

Usage:
    python mvdb_glue_profile.py [--rows 10000] [--queries 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)


def vector_literal(values):
    return "[" + ",".join("%.6f" % value for value in values) + "]"


class Clock:
    """Accumulated seconds per label."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def add(self, label, seconds):
        self.totals[label] = self.totals.get(label, 0.0) + seconds
        self.counts[label] = self.counts.get(label, 0) + 1

    def report(self, queries, label_width=22):
        total = sum(self.totals.values())
        print("  %-*s %10s %8s" % (label_width, "stage", "us/query", "share"))
        for label in sorted(self.totals, key=self.totals.get, reverse=True):
            per = self.totals[label] / queries * 1e6
            print("  %-*s %10.2f %7.1f%%"
                  % (label_width, label, per, per / (total / queries * 1e6) * 100))
        print("  %-*s %10.2f" % (label_width, "measured total",
                                 total / queries * 1e6))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)

    from minivectordb.engine import Engine
    from minivectordb import fastlane, fastpath
    import minivectordb.engine as engine_module

    rng = random.Random(args.seed)
    root = tempfile.mkdtemp(prefix="mvdb-glue-",
                            dir=os.path.join(os.path.dirname(REPO), "tmp"))
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    engine = Engine(data_dir)
    engine.open()

    print("fastlane available: %s | numpy %s"
          % (fastlane.available(), fastpath.module().__version__))

    try:
        engine.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, "
                       "category VARCHAR(16), content VARCHAR(64), "
                       "embedding VECTOR(%d));" % args.dim)
        for offset in range(0, args.rows, 100):
            values = []
            for index in range(offset, min(offset + 100, args.rows)):
                vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
                values.append("(%d,'cat-%d','payload-%d','%s')"
                              % (index + 1, index % args.categories, index,
                                 vector_literal(vector)))
            engine.execute("INSERT INTO docs VALUES %s;" % ",".join(values))

        statements = []
        for index in range(args.queries):
            vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
            if index % 2 == 0:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance "
                       "FROM docs WHERE category = 'cat-%d' ORDER BY distance "
                       "LIMIT %d;" % (vector_literal(vector),
                                       index % args.categories, args.k))
            else:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance "
                       "FROM docs ORDER BY distance LIMIT %d;"
                       % (vector_literal(vector), args.k))
            statements.append(sql)

        # ---- wrap the pieces the engine is made of --------------------
        clock = Clock()
        band_sizes = []
        original_topk = fastlane.topk
        original_parse = fastlane.parse_text
        original_scan = engine_module.Engine._scan_topk
        original_plan = engine_module.Engine._fast_plan

        def timed_topk(*args_, **kwargs_):
            start = time.perf_counter()
            result = original_topk(*args_, **kwargs_)
            clock.add("kernel topk", time.perf_counter() - start)
            if result is not None:
                band_sizes.append(int(result[0].size))
            return result

        def timed_parse(*args_, **kwargs_):
            start = time.perf_counter()
            result = original_parse(*args_, **kwargs_)
            clock.add("C literal parse", time.perf_counter() - start)
            return result

        def timed_scan(self, *args_, **kwargs_):
            start = time.perf_counter()
            result = original_scan(self, *args_, **kwargs_)
            clock.add("scan (total)", time.perf_counter() - start)
            return result

        def timed_plan(self, *args_, **kwargs_):
            start = time.perf_counter()
            result = original_plan(self, *args_, **kwargs_)
            clock.add("fast plan", time.perf_counter() - start)
            return result

        fastlane.topk = timed_topk
        fastlane.parse_text = timed_parse
        engine_module.Engine._scan_topk = timed_scan
        engine_module.Engine._fast_plan = timed_plan

        for sql in statements[:50]:
            engine.execute(sql, "exact")

        clock.totals.clear()
        clock.counts.clear()
        band_sizes.clear()

        whole = time.perf_counter()
        for sql in statements:
            engine.execute(sql, "exact")
        whole = time.perf_counter() - whole

        fastlane.topk = original_topk
        fastlane.parse_text = original_parse
        engine_module.Engine._scan_topk = original_scan
        engine_module.Engine._fast_plan = original_plan

        print()
        print("=== engine.execute 的真实路径分解 (queries=%d) ===" % args.queries)
        clock.report(args.queries)
        print()
        print("  whole execute      : %8.2f us/query"
              % (whole / args.queries * 1e6))
        if band_sizes:
            band_sizes.sort()
            print("  kernel band size   : median %d, max %d"
                  % (band_sizes[len(band_sizes) // 2], band_sizes[-1]))
    finally:
        try:
            engine.close()
        except Exception:
            pass
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
