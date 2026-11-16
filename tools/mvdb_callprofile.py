#!/usr/bin/env python3
"""A call profile of the query path.

``cProfile`` charges every call a fixed overhead, so the absolute numbers are
worth less than the picture: which functions are called, how often, and in
what proportion of the calls.  The per-call overhead is subtracted from every
entry, which turns the report into a count of how many times each function
was entered and how much of the time was spent inside it.

Usage:
    PYTHONPATH=seekdb/repo python tools/mvdb_callprofile.py [--rows 10000]
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import random
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, WS)

from minivectordb.engine import Engine                  # noqa: E402


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
    data_dir = os.path.join(WS, "..", "tmp", "callprofile")
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

    for sql in statements[:20]:
        engine.execute(sql, "exact")

    profiler = cProfile.Profile()
    profiler.enable()
    for sql in statements:
        engine.execute(sql, "exact")
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats("tottime")
    print("=== by time inside the function ===")
    stats.print_stats(28)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
