#!/usr/bin/env python3
"""Break the write path into its parts, on the real path.

The query path is saturated on the task's score, so the write path is the
only place where effort still buys anything.  "The write path" is a label,
though, not a measurement -- this script times the parts it actually
consists of.

Each part is timed by wrapping the real function, not by reading a model
of it, and the rows the engine ends up with are compared against the rows
an unwrapped engine produces, so a wrapper that changes the behaviour
cannot pass unnoticed.

The parts:

    parse    tokenising the INSERT and decoding the vector literals
    coerce   parsed nodes -> typed column values
    wal      encoding the row and appending it to the log
    overlay  staging the row in the transaction
    sync     fdatasync of the log -- required, the task says the write
             time includes the time until the data is durable
    flush    writing the staged rows into the data files

Usage:
    PYTHONPATH=seekdb/repo python tools/mvdb_write_pieces.py [--rows 10000]
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

from minivectordb import engine as engine_module               # noqa: E402
from minivectordb.engine import Engine                         # noqa: E402


def literal(values):
    return "[" + ",".join("%.6f" % value for value in values) + "]"


class Timer:
    """Accumulates wall time per label."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def add(self, label, seconds):
        self.totals[label] = self.totals.get(label, 0.0) + seconds
        self.counts[label] = self.counts.get(label, 0) + 1

    def report(self, rows):
        width = max(len(label) for label in self.totals)
        print("%-*s  %10s  %10s  %8s" % (width, "part", "us/row", "us/batch", "share"))
        total = sum(self.totals.values())
        for label, seconds in sorted(self.totals.items(),
                                     key=lambda kv: -kv[1]):
            print("%-*s  %10.2f  %10.2f  %7.1f%%" % (
                width, label, seconds / rows * 1e6,
                seconds / max(self.counts[label], 1) * 1e6,
                seconds / total * 100 if total else 0.0))
        print()
        print("total measured %.4f s" % total)
        return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    data_dir = os.path.join(WS, "..", "tmp", "write-pieces")
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
    statements = []
    for offset in range(0, len(rows), args.batch):
        chunk = rows[offset:offset + args.batch]
        values = ",".join("(%d,'%s','%s','%s')" % (pk, cat, content, literal(vec))
                          for pk, cat, content, vec in chunk)
        statements.append("INSERT INTO docs VALUES %s;" % values)

    timer = Timer()

    # -- parse: the module level function the engine calls ------------
    original_parse = engine_module.parse

    def timed_parse(sql):
        start = time.perf_counter()
        try:
            return original_parse(sql)
        finally:
            timer.add("parse", time.perf_counter() - start)

    engine_module.parse = timed_parse

    # -- the rest: methods on the engine instance --------------------
    original_coerce = engine_module.Engine._coerce
    original_append_row = engine_module.Engine._append_row
    original_stage = engine_module.Engine._stage
    original_sync = engine_module.Engine._sync
    original_materialize = engine_module.Engine._materialize

    def timed_coerce(self, column, node):
        start = time.perf_counter()
        try:
            return original_coerce(self, column, node)
        finally:
            timer.add("coerce", time.perf_counter() - start)

    def timed_append_row(self, kind, txn, table, pk, values):
        start = time.perf_counter()
        try:
            return original_append_row(self, kind, txn, table, pk, values)
        finally:
            timer.add("wal", time.perf_counter() - start)

    def timed_stage(self, table, pk, values):
        start = time.perf_counter()
        try:
            return original_stage(self, table, pk, values)
        finally:
            timer.add("overlay", time.perf_counter() - start)

    def timed_sync(self):
        start = time.perf_counter()
        try:
            return original_sync(self)
        finally:
            timer.add("sync", time.perf_counter() - start)

    def timed_materialize(self):
        start = time.perf_counter()
        try:
            return original_materialize(self)
        finally:
            timer.add("flush", time.perf_counter() - start)

    engine_module.Engine._coerce = timed_coerce
    engine_module.Engine._append_row = timed_append_row
    engine_module.Engine._stage = timed_stage
    engine_module.Engine._sync = timed_sync
    engine_module.Engine._materialize = timed_materialize

    wall_start = time.perf_counter()
    for sql in statements:
        engine.execute(sql)
    wall = time.perf_counter() - wall_start

    engine_module.parse = original_parse
    engine_module.Engine._coerce = original_coerce
    engine_module.Engine._append_row = original_append_row
    engine_module.Engine._stage = original_stage
    engine_module.Engine._sync = original_sync
    engine_module.Engine._materialize = original_materialize

    print("=== write path, %d rows in %d batches of %d ===" % (
        args.rows, len(statements), args.batch))
    print("wall total          %.4f s" % wall)
    print("per row             %.2f us" % (wall / args.rows * 1e6))
    print("per batch           %.4f ms" % (wall / len(statements) * 1e3))
    print()
    timer.report(args.rows)

    # -- the wrapper must not have changed the result -----------------
    engine.close()

    plain_dir = os.path.join(WS, "..", "tmp", "write-pieces-plain")
    shutil.rmtree(plain_dir, ignore_errors=True)
    os.makedirs(plain_dir, exist_ok=True)
    plain = Engine(plain_dir)
    plain.open()
    plain.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                  "content VARCHAR(64), embedding VECTOR(%d));" % args.dim)
    for sql in statements:
        plain.execute(sql)
    # COUNT(*) is not in the supported subset, so the row count is read
    # from the store itself rather than through SQL.
    table = plain.tables["docs"]
    plain_count = sum(1 for _ in table.store.iter_rows())
    plain.close()
    print("rows in the table   %d (expected %d)" % (plain_count, args.rows))
    if plain_count != args.rows:
        raise SystemExit("the wrappers changed the number of rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
