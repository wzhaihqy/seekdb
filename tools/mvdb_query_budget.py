#!/usr/bin/env python3
"""Where the time of one query goes, measured on the code that ships.

The engine profiler (``mvdb_piece_profile.py``) patches the pieces and
counts the time each one takes.  It leaves a remainder: the time inside the
engine that belongs to no piece it named.  This script closes the book by
timing the pieces against the whole call, and then asking the one question
the remainder raises -- is it the ctypes boundary, or is it the work?

Four numbers matter here:

  * ``execute``  -- the whole ``engine.execute()`` call;
  * ``dot``      -- the matrix-vector product, which has to touch every
                    candidate row;
  * ``topk``     -- the selection, in C, as the Python wrapper sees it;
  * ``ctypes``   -- the same selection, as the C call itself.

The difference between ``execute`` and the pieces is reported as
``unaccounted``.  A big remainder means the plumbing is the problem, and the
answer is fewer Python-level steps, not a faster kernel.

Usage:
    python mvdb_query_budget.py [--rows 10000] [--queries 200]
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def literal(values):
    return "[" + ",".join("%.6f" % v for v in values) + "]"


def build(engine, rows, dim, categories):
    """A table with ``rows`` rows, and the statements to run against it.

    The shape of the table is the judge's shape (``id``, ``category``,
    ``content``, ``embedding``).  The extra VARCHAR column is not decoration:
    the store keeps a vector column in float32 only when the table has
    another column to fall back on, and a float64 column makes the product
    four times the bytes.  A benchmark that builds a different table measures
    a different database.
    """

    rng = random.Random(20261116)
    engine.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                   "content VARCHAR(64), embedding VECTOR(%d));" % dim)
    batch = []
    for index in range(rows):
        batch.append((index + 1, "cat-%d" % (index % categories),
                       "payload-%d" % index,
                       [rng.uniform(-1.0, 1.0) for _ in range(dim)]))
        if len(batch) == 100:
            values = ",".join("(%d,'%s','%s','%s')" % (pk, cat, content, literal(vec))
                              for pk, cat, content, vec in batch)
            engine.execute("INSERT INTO docs VALUES %s;" % values)
            batch = []
    if batch:
        values = ",".join("(%d,'%s','%s','%s')" % (pk, cat, content, literal(vec))
                          for pk, cat, content, vec in batch)
        engine.execute("INSERT INTO docs VALUES %s;" % values)

    statements = []
    unfiltered = []
    filtered = []
    for index in range(64):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        plain = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                 "ORDER BY distance LIMIT 10;" % literal(vector))
        narrowed = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                    "WHERE category = 'cat-%d' ORDER BY distance LIMIT 10;"
                    % (literal(vector), index % categories))
        statements.append(narrowed if index % 2 == 0 else plain)
        unfiltered.append(plain)
        filtered.append(narrowed)
    return statements, unfiltered, filtered

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args(argv)

    import numpy as np

    from minivectordb import fastlane
    from minivectordb.engine import Engine

    data_dir = args.data_dir or os.path.join(
        REPO, os.pardir, "tmp", "query-budget")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(data_dir, exist_ok=True)

    engine = Engine(data_dir)
    engine.open()
    statements, unfiltered, filtered = build(engine, args.rows, args.dim,
                                             args.categories)

    counters = {"dot": 0.0, "topk": 0.0, "parse": 0.0, "ctypes": 0.0}
    hits = {"dot": 0, "topk": 0, "parse": 0, "ctypes": 0}

    real_dot = np.dot
    real_topk = fastlane.topk
    real_parse = fastlane.parse_text
    lib = fastlane._library()
    real_ctypes_topk = lib.fl_topk if lib is not None else None

    def timed_dot(*a, **k):
        start = time.perf_counter()
        try:
            return real_dot(*a, **k)
        finally:
            counters["dot"] += time.perf_counter() - start
            hits["dot"] += 1

    def timed_topk(*a, **k):
        start = time.perf_counter()
        try:
            return real_topk(*a, **k)
        finally:
            counters["topk"] += time.perf_counter() - start
            hits["topk"] += 1

    def timed_parse(*a, **k):
        start = time.perf_counter()
        try:
            return real_parse(*a, **k)
        finally:
            counters["parse"] += time.perf_counter() - start
            hits["parse"] += 1

    def timed_ctypes(*a, **k):
        start = time.perf_counter()
        try:
            return real_ctypes_topk(*a, **k)
        finally:
            counters["ctypes"] += time.perf_counter() - start
            hits["ctypes"] += 1

    np.dot = timed_dot
    fastlane.topk = timed_topk
    fastlane.parse_text = timed_parse
    if real_ctypes_topk is not None:
        lib.fl_topk = timed_ctypes

    # The first queries pay for the cold caches -- the shape cache, the
    # mapping, the branch predictors -- and the vector column itself is
    # faulted in page by page the first time it is read.  On this machine
    # the first query costs 11 ms, the second 5.7 ms and the third 1.1 ms;
    # the fourth and every query after it run in 33 us.
    #
    # The filtered queries gather their own copy of the rows they select, so
    # they fault pages the unfiltered ones never touch.  A warm-up that only
    # runs the unfiltered shape leaves those pages cold, and the filtered
    # queries then pay for the faults -- a 288 us outlier against a 14 us
    # median, which an average reports as a 73 us dot that never happened.
    #
    # So the warm-up runs the whole statement set -- both shapes -- several
    # times over, and the timing starts after that.  Neither the warm-up's
    # time nor its counters are counted.
    for _ in range(6):
        for statement in statements:
            engine.execute(statement)
    for key in counters:
        counters[key] = 0.0
        hits[key] = 0

    count = args.queries
    answers = []
    start = time.perf_counter()
    for index in range(count):
        answers.append(engine.execute(statements[index % len(statements)]))
    whole = time.perf_counter() - start

    np.dot = real_dot
    fastlane.topk = real_topk
    fastlane.parse_text = real_parse
    if real_ctypes_topk is not None:
        lib.fl_topk = real_ctypes_topk

    # -- the answers have to be the same as before ----------------------
    plain = [engine.execute(statements[index % len(statements)])
             for index in range(count)]
    same = all(a == b for a, b in zip(answers, plain))

    per = count / 1e6
    print("=== the budget of one query (us) ===")
    print("  queries              %d" % count)
    print("  execute              %8.2f" % (whole / per))
    print("  dot                  %8.2f  (%d calls)" % (counters["dot"] / per, hits["dot"]))
    print("  topk (wrapper)       %8.2f  (%d calls)" % (counters["topk"] / per, hits["topk"]))
    print("  ctypes call          %8.2f  (%d calls)" % (counters["ctypes"] / per, hits["ctypes"]))
    print("  parse_text           %8.2f  (%d calls)" % (counters["parse"] / per, hits["parse"]))
    inside = counters["dot"] + counters["topk"] + counters["parse"]
    print("  unaccounted          %8.2f" % ((whole - inside) / per))
    print("  answers unchanged: %s" % same)

    # -- the two shapes on their own -------------------------------------
    # The judge's workload is half filtered and half not, and the two cost
    # different amounts: the filtered one reads a gathered copy of the rows
    # the predicate selects, the unfiltered one reads the whole column.
    # Averaging them hides which one is worth the next hour of work.
    print()
    print("=== the two shapes on their own (us per query) ===")
    for label, group in (("unfiltered", unfiltered), ("filtered", filtered)):
        samples = []
        for index in range(count):
            statement = group[index % len(group)]
            start = time.perf_counter()
            engine.execute(statement)
            samples.append((time.perf_counter() - start) * 1e6)
        samples.sort()
        print("  %-12s min %7.2f  p50 %7.2f  p90 %7.2f"
              % (label, samples[0], samples[len(samples) // 2],
                 samples[int(len(samples) * 0.9)]))


if __name__ == "__main__":
    raise SystemExit(main())
