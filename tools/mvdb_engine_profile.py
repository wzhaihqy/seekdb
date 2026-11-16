#!/usr/bin/env python3
"""Where the 53 us of one engine call goes, inside the engine.

``mvdb_loop_profile.py`` says the server spends 53 us inside
``engine.execute`` per query.  This tool says what that 53 us is made of.

It drives the real engine -- the real store, the real memory mapping, the
real filter cache, the real kernel -- and times the steps the fast path is
made of, one query at a time, on the workload's own shape (10,000 rows of
128 dimensions, K = 10, half of the queries carrying an equality filter).

The stages are:

    plan      the statement is parsed and the shape is recognised
    literal   the vector literal is turned into float32
    scan      the column is read and the coarse product is taken
    select    the closest rows are chosen
    exact     the survivors get their float64 distances
    project   the answer rows are built
    json      the response is encoded

Every stage is timed inside the engine process, so the numbers include the
memory traffic of the memory mapping and not the traffic of a fresh array.

Usage:
    python mvdb_engine_profile.py [--rows 10000] [--queries 1000]
                                  [--dim 128] [--k 10] [--categories 8]
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy                                            # noqa: E402
from minivectordb import fastlane, fastpath             # noqa: E402
from minivectordb.engine import Engine                  # noqa: E402


def vector_literal(values):
    return "[" + ",".join("%.6f" % value for value in values) + "]"


class Timer:
    """A set of named accumulators, measured with the clock."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def add(self, name, seconds):
        self.totals[name] = self.totals.get(name, 0.0) + seconds
        self.counts[name] = self.counts.get(name, 0) + 1

    def report(self, queries):
        print("=== 引擎内部一次查询的分解 (queries=%d) ===" % queries)
        print("  %-10s %10s %8s" % ("stage", "us/query", "share"))
        total = sum(self.totals.values())
        for name in sorted(self.totals, key=lambda n: -self.totals[n]):
            value = self.totals[name] / queries * 1e6
            share = (self.totals[name] / total * 100) if total else 0.0
            print("  %-10s %10.1f %7.1f%%" % (name, value, share))
        print("  %-10s %10.1f" % ("total", total / queries * 1e6))
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)

    root = os.environ.get("MVDB_SCRATCH") or os.path.join(
        os.path.dirname(REPO), "tmp", "engine-profile")
    os.makedirs(root, exist_ok=True)
    data_dir = tempfile.mkdtemp(prefix="mvdb-engine-", dir=root)

    rng = random.Random(args.seed)
    engine = Engine(data_dir)
    engine.open()
    engine.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                   "content VARCHAR(64), embedding VECTOR(%d));" % args.dim)
    for offset in range(0, args.rows, 100):
        values = ",".join(
            "(%d,'cat-%d','row-%d','%s')"
            % (pk, pk % args.categories, pk,
               vector_literal([rng.uniform(-1.0, 1.0) for _ in range(args.dim)]))
            for pk in range(offset + 1, min(offset + 101, args.rows + 1)))
        engine.execute("INSERT INTO docs VALUES %s;" % values)
    engine.execute("CREATE VECTOR INDEX docs_vec ON docs(embedding);")

    print("fastlane available:", fastlane.available(),
          "| numpy", numpy.__version__)
    print("rows:", args.rows, "dim:", args.dim, "k:", args.k)
    print()

    # ---- the engine's own stages, measured by wrapping the pieces -------
    timer = Timer()

    table = engine.tables.get("docs")
    store = table.store

    queries = []
    for index in range(args.queries):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
        if index % 2 == 0:
            sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                   "ORDER BY distance LIMIT %d;" % (vector_literal(vector), args.k))
        else:
            sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                   "WHERE category = 'cat-%d' ORDER BY distance LIMIT %d;"
                   % (vector_literal(vector), index % args.categories, args.k))
        queries.append(sql)

    # Warm the caches the way the steady state sees them.
    for sql in queries[:50]:
        engine.execute(sql)

    # -- one query at a time, with the stages timed ----------------------
    from minivectordb.core import to_float32_literal     # noqa: E402
    from minivectordb.sqlparser import parse              # noqa: E402

    for sql in queries:
        t0 = time.perf_counter()
        statement = parse(sql)
        t1 = time.perf_counter()
        timer.add("parse", t1 - t0)

        # The plan is the engine's own: it knows where the vector column is
        # and which item carries the distance.
        plan = engine._fast_plan(table, statement)
        column_position = plan[0]
        item_index = plan[8]
        literal_at = plan[9]

        # The vector literal is the argument the plan points at.
        if item_index >= 0:
            node = statement["items"][item_index]["expr"]
        else:
            node = statement["order"][0][0]
        argument = node[2][literal_at][1]
        query32 = numpy.asarray(to_float32_literal(argument), dtype=numpy.float32)
        t2 = time.perf_counter()
        timer.add("literal", t2 - t1)

        scanned = engine._scan_topk(table, column_position, query32,
                                    statement["where"], args.k)
        t3 = time.perf_counter()
        timer.add("scan", t3 - t2)

        slots = scanned[0]
        view = engine._view(store, column_position)
        matrix = view[0]
        exact = fastlane.exact_distances(matrix[slots], query32)
        t4 = time.perf_counter()
        timer.add("exact", t4 - t3)

        order = numpy.lexsort((view[2][slots], exact))
        chosen = order[:args.k]
        rows = store.read_slots_columns(slots[chosen].tolist(), [1, 2])
        output = [[rows[i][0], float(exact[chosen[i]])] for i in range(len(chosen))]
        t5 = time.perf_counter()
        timer.add("project", t5 - t4)

        import json
        json.dumps({"id": 1, "ok": True, "columns": ["id", "distance"],
                    "rows": output}, ensure_ascii=True, allow_nan=False,
                   separators=(",", ":"))
        t6 = time.perf_counter()
        timer.add("json", t6 - t5)

    timer.report(len(queries))

    # ---- the two query shapes, measured apart --------------------------
    for label, predicate in (("no filter", None),
                             ("with filter", "category = 'cat-3'")):
        samples = []
        for index in range(200):
            vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
            if predicate is None:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                       "ORDER BY distance LIMIT %d;" % (vector_literal(vector), args.k))
            else:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                       "WHERE %s ORDER BY distance LIMIT %d;"
                       % (vector_literal(vector), predicate, args.k))
            start = time.perf_counter()
            engine.execute(sql)
            samples.append((time.perf_counter() - start) * 1e6)
        samples.sort()
        print("  %-12s median %7.1f us   min %7.1f us   max %7.1f us"
              % (label, samples[len(samples) // 2], samples[0], samples[-1]))

    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
