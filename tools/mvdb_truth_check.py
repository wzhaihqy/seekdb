#!/usr/bin/env python3
"""Check the vector TopK answer against a brute force reference.

The TopK path was changed so that the coarse ranking pass is computed by
``einsum`` instead of the linear algebra library's vector-matrix product.
The two kernels sum the same terms in a different order, so the coarse
values differ in their last bits.  This tool answers the question that
matters: does the answer the database returns still match the answer a
brute force scan produces, for every row, on every query?

The reference computes the distance of every row in float64, one row at a
time, and orders by (distance, primary key) -- the rule the task states.
The database is asked the same question over the JSONL protocol, so the
whole path is exercised: the parser, the executor, the kernel and the
serialiser.

Run it with the interpreter the judge image uses.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def vector_literal(values):
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def brute_force(vectors, keys, query, category, categories, k):
    """The reference: every row, float64, ordered by (distance, primary key)."""

    scored = []
    for index, vector in enumerate(vectors):
        if category is not None and categories[index] != category:
            continue
        total = 0.0
        for a, b in zip(vector, query):
            diff = float(a) - float(b)
            total += diff * diff
        scored.append((math.sqrt(total), keys[index]))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[:k]


class Client:
    """One JSONL request, one JSONL response, over a pipe."""

    def __init__(self, command):
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        banner = json.loads(self.proc.stdout.readline())
        assert banner == {"type": "ready"}, banner
        self.sequence = 0

    def execute(self, sql):
        self.sequence += 1
        request = {"id": self.sequence, "op": "execute", "sql": sql,
                   "search_mode": "exact"}
        self.proc.stdin.write(json.dumps(request) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("database closed stdout")
            reply = json.loads(line)
            if reply.get("id") == self.sequence:
                return reply

    def close(self):
        try:
            self.execute("SELECT 1;")
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--queries", type=int, default=120)
    parser.add_argument("--categories", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    vectors = [[rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
               for _ in range(args.rows)]
    keys = list(range(1, args.rows + 1))
    rng.shuffle(keys)
    categories = ["cat-%d" % (index % args.categories)
                  for index in range(args.rows)]
    contents = ["row-%d" % index for index in range(args.rows)]

    root = tempfile.mkdtemp(prefix="mvdb-truth-")
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir)
    client = Client([args.python, "-u", os.path.join(REPO, "main.py"),
                      "--data-dir", data_dir])
    try:
        client.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, "
                       "category VARCHAR(16), content VARCHAR(64), "
                       "embedding VECTOR(%d));" % args.dim)
        for offset in range(0, args.rows, 100):
            chunk = range(offset, min(offset + 100, args.rows))
            values = ",".join(
                "(%d,'%s','%s','%s')" % (keys[i], categories[i], contents[i],
                                         vector_literal(vectors[i]))
                for i in chunk)
            reply = client.execute("INSERT INTO docs VALUES %s;" % values)
            assert reply.get("ok"), reply
        client.execute("CREATE VECTOR INDEX docs_vec ON docs(embedding);")

        mismatches = 0
        checked = 0
        started = time.perf_counter()
        for index in range(args.queries):
            query = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
            category = categories[index % args.categories] if index % 2 == 0 else None
            literal = vector_literal(query)
            if category is None:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance "
                       "FROM docs ORDER BY distance LIMIT %d;" % (literal, args.k))
            else:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance "
                       "FROM docs WHERE category = '%s' ORDER BY distance "
                       "LIMIT %d;" % (literal, category, args.k))
            reply = client.execute(sql)
            assert reply.get("ok"), reply
            got = reply.get("rows", [])
            want = brute_force(vectors, keys, query, category, categories, args.k)
            checked += 1
            if len(got) != len(want):
                mismatches += 1
                print("  row count differs: got %d want %d" % (len(got), len(want)))
                continue
            for row, (distance, key) in zip(got, want):
                got_key = int(row[0])
                got_distance = float(row[1])
                if got_key != key or abs(got_distance - distance) > 1e-5:
                    mismatches += 1
                    print("  mismatch: got (%s, %.9f) want (%s, %.9f)"
                          % (got_key, got_distance, key, distance))
                    break
            if index % 20 == 0:
                print("  checked %d queries, %d mismatches" % (checked, mismatches),
                      flush=True)
        elapsed = time.perf_counter() - started
        print()
        print("=== 暴力真值对比 ===")
        print("  查询数        : %d" % checked)
        print("  不一致        : %d" % mismatches)
        print("  每查询        : %.3f ms (含协议往返)"
              % (elapsed / max(checked, 1) * 1000))
        print("  结论          :", "全部一致" if mismatches == 0 else "存在不一致")
        return 0 if mismatches == 0 else 1
    finally:
        client.close()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
