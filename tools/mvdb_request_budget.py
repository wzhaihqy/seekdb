#!/usr/bin/env python3
"""What one request costs the server, section by section.

The bench measures a query from the outside: the client writes a line, the
server reads it, parses it, runs it, writes the answer back.  Half of that
time is the round trip and half is the engine, and the two have different
owners -- the round trip belongs to the pipe and to the JSON, the engine to
the code.  A score is made of both, so both have to be measured.

This script takes the bytes the judge actually sends and times each stage
on its own:

  read        the cost of getting one line out of the pipe
  json.loads  the cost of turning the request line into a dict
  execute     the cost of running the statement
  json.dumps  the cost of turning the answer into a line
  write       the cost of writing the line out

The statement is the one the judge sends: a TopK query over the judge's
table, with the judge's vector literal.  The two shapes are measured
apart, because a filtered query gathers its own rows and a whole table
query does not.

Usage:
    PYTHONPATH=seekdb/repo python tools/mvdb_request_budget.py [--rows 10000]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
sys.path.insert(0, WS)

import numpy as np                                        # noqa: E402

from minivectordb.engine import Engine                    # noqa: E402


def vector_literal(values):
    return "[" + ",".join("%.6f" % value for value in values) + "]"


def build_dataset(rows, dim, categories, rng):
    data = []
    for index in range(rows):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        data.append((index + 1, "cat-%d" % (index % categories),
                     "payload-%d" % index, vector))
    return data


def best_of(fn, rounds=9, inner=20):
    """The smallest average of ``inner`` calls, over ``rounds`` rounds."""

    out = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        out.append((time.perf_counter() - start) / inner)
    return min(out) * 1e6


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args(argv)

    root = os.path.join(os.path.dirname(WS), "tmp", "reqbudget")
    os.makedirs(root, exist_ok=True)
    data_dir = os.path.join(root, "data")
    os.system("rm -rf %s" % data_dir)
    os.makedirs(data_dir, exist_ok=True)

    rng = random.Random(args.seed)
    data = build_dataset(args.rows, args.dim, args.categories, rng)

    engine = Engine(data_dir)
    engine.open()
    engine.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, "
                   "category VARCHAR(16), content VARCHAR(64), "
                   "embedding VECTOR(%d));" % args.dim)
    for offset in range(0, len(data), 100):
        chunk = data[offset:offset + 100]
        values = ",".join("(%d,'%s','%s','%s')"
                          % (pk, category, content, vector_literal(vector))
                          for pk, category, content, vector in chunk)
        engine.execute("INSERT INTO docs VALUES %s;" % values)
    engine.execute("CREATE VECTOR INDEX docs_vec ON docs(embedding);")

    shapes = {"unfiltered": [], "filtered": []}
    for index in range(40):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
        literal = vector_literal(vector)
        shapes["unfiltered"].append(
            "SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
            "ORDER BY distance LIMIT 10;" % literal)
        shapes["filtered"].append(
            "SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
            "WHERE category = 'cat-%d' ORDER BY distance LIMIT 10;"
            % (literal, index % args.categories))

    print("=== the cost of one request, stage by stage (us) ===")
    print("%-11s %8s %8s %8s %8s %8s %8s" %
          ("shape", "read", "loads", "execute", "dumps", "write", "sum"))

    for label, statements in shapes.items():
        # The line the judge sends, byte for byte.
        lines = []
        for index, sql in enumerate(statements):
            request = {"id": index + 1, "op": "execute", "sql": sql,
                       "search_mode": "exact"}
            lines.append((json.dumps(request, ensure_ascii=False)
                          + "\n").encode("utf-8"))

        # The read cost: one line out of a buffered stream.  The stream is a
        # file rather than a pipe, because the pipe cannot be rewound and
        # the lines have to be there before the first read.  The line the
        # server reads is the same either way.
        path = os.path.join(root, "requests.jsonl")
        with open(path, "wb") as handle:
            handle.write(b"".join(lines))
        reader = io.open(path, "rb")

        def read_one():
            return reader.readline()

        read_cost = best_of(read_one, rounds=9, inner=len(lines))

        def loads_one(line=lines[0]):
            return json.loads(line)

        loads_cost = best_of(loads_one)

        def execute_one(sql=statements[0]):
            return engine.execute(sql)

        execute_cost = best_of(execute_one)

        answer = engine.execute(statements[0])
        payload = {"id": 1, "ok": True}
        payload.update(answer)

        def dumps_one():
            return json.dumps(payload, ensure_ascii=True, allow_nan=False,
                              separators=(",", ":"))

        dumps_cost = best_of(dumps_one)

        sink = io.BytesIO()

        def write_one():
            sink.write(dumps_one().encode("ascii") + b"\n")

        write_cost = best_of(write_one)

        total = read_cost + loads_cost + execute_cost + dumps_cost + write_cost
        print("%-11s %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (label, read_cost, loads_cost, execute_cost, dumps_cost,
                 write_cost, total))

    engine.close()
    print()
    print("The row a request answers with is small; the request itself is")
    print("not, because the vector literal is 128 numbers of text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
