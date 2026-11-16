#!/usr/bin/env python3
"""The glue of one query, one piece at a time.

The engine's own profile says a query costs about eleven microseconds that
neither the product nor the kernel accounts for.  This script measures the
pieces of that eleven by calling them one at a time on the same statement,
so the leftover can be looked at.

Usage:
    PYTHONPATH=seekdb/repo python tools/mvdb_glue2.py [--rows 10000]
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

from minivectordb import fastlane, fastpath             # noqa: E402
from minivectordb.engine import Engine                  # noqa: E402
from minivectordb.sqlparser import parse                # noqa: E402


def best_of(fn, rounds=9, inner=30):
    out = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        out.append((time.perf_counter() - start) / inner)
    return min(out) * 1e6


def literal(values):
    return "[" + ",".join("%.6f" % v for v in values) + "]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    data_dir = os.path.join(WS, "..", "tmp", "glue2")
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
        values = ",".join("(%d,'%s','%s','%s')" % (pk, cat, content,
                                                    literal(vec))
                          for pk, cat, content, vec in chunk)
        engine.execute("INSERT INTO docs VALUES %s;" % values)

    qv = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
    sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
           "ORDER BY distance LIMIT 10;" % literal(qv))
    sql_filtered = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                    "WHERE category = 'cat-0' ORDER BY distance LIMIT 10;" % literal(qv))

    statement = parse(sql)
    print("statement keys:", sorted(statement))

    print("\n=== the statement itself ===")
    print("  %-40s %8.2f us" % ("parse(sql)", best_of(lambda: parse(sql))))
    print("  %-40s %8.2f us" % ("parse(sql, filtered)",
                                best_of(lambda: parse(sql_filtered))))

    node = statement["items"][1]["expr"]
    argument = node[2][1][1]
    print("\n=== the vector literal ===")
    print("  %-40s %8.2f us" % ("fastlane.parse_text", best_of(
        lambda: fastlane.parse_text(argument, args.dim))))
    query32 = fastlane.parse_text(argument, args.dim)
    print("  %-40s %8.2f us" % ("query32.astype(float64)", best_of(
        lambda: query32.astype(np.float64))))

    print("\n=== the plan ===")
    table = engine.tables["docs"]
    print("  %-40s %8.2f us" % ("_fast_plan", best_of(
        lambda: engine._fast_plan(table, statement))))

    print("\n=== the whole call ===")
    print("  %-40s %8.2f us" % ("engine.execute(sql)", best_of(
        lambda: engine.execute(sql, "exact"), inner=20)))
    print("  %-40s %8.2f us" % ("engine.execute(sql_filtered)", best_of(
        lambda: engine.execute(sql_filtered, "exact"), inner=20)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
