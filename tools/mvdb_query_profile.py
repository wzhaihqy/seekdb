"""Where does the fixed cost of a query go?

The benchmark shows that a query over ten thousand rows costs about twice as
much as a query over a hundred rows, so most of the time is not the scan but
a fixed cost per statement.  This script profiles the query phase with
``cProfile`` and prints the functions with the highest self time, so the
optimisation target is measured instead of guessed.

Usage:
    python tools/mvdb_query_profile.py [--rows 10000] [--queries 200]
"""
from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import random
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, WS)

from minivectordb.engine import Engine  # noqa: E402
from minivectordb.sqlparser import parse  # noqa: E402


def build(rows, dim, categories, rng):
    data = []
    for index in range(rows):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        data.append((index + 1, "cat-%d" % (index % categories),
                     "payload-%d" % index, vector))
    return data


def literal(values):
    return "[" + ",".join("%.6f" % value for value in values) + "…".replace("…", "") + "]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    data = build(args.rows, args.dim, args.categories, rng)
    queries = []
    for index in range(args.queries):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
        category = "cat-%d" % (index % args.categories) if index % 2 == 0 else None
        queries.append((vector, category))

    root = tempfile.mkdtemp(prefix="mvdb-profile-")
    try:
        data_dir = os.path.join(root, "data")
        os.makedirs(data_dir, exist_ok=True)
        engine = Engine(data_dir)
        engine.open()
        engine.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                       "content VARCHAR(64), embedding VECTOR(%d));" % args.dim)
        for start in range(0, len(data), args.batch):
            chunk = data[start:start + args.batch]
            values = ",".join("(%d,'%s','%s','%s')" % (row[0], row[1], row[2], literal(row[3]))
                              for row in chunk)
            engine.execute("INSERT INTO docs VALUES %s;" % values)
        engine.execute("CREATE VECTOR INDEX docs_vec ON docs(embedding);")

        statements = []
        for vector, category in queries:
            if category is None:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                       "ORDER BY distance LIMIT 10;" % literal(vector))
            else:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                       "WHERE category = '%s' ORDER BY distance LIMIT 10;"
                       % (literal(vector), category))
            statements.append(sql)

        start = time.perf_counter()
        for sql in statements:
            engine.execute(sql)
        plain = time.perf_counter() - start

        profiler = cProfile.Profile()
        profiler.enable()
        for sql in statements:
            engine.execute(sql)
        profiler.disable()
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(25)
        print("queries: %d, rows: %d, plain %.3f ms per query"
              % (len(statements), args.rows, plain / len(statements) * 1000))
        print(stream.getvalue())
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
