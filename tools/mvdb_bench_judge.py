#!/usr/bin/env python3
"""Benchmark the server the way the judge runs it.

The judge gives the container ``--cpus=2``: two CPUs of runtime, and the
affinity mask still shows every core of the machine.  A twenty core
workstation does not reproduce that -- a linear algebra library opens a
worker per visible core, the workers share two CPUs, and the mean the
score uses is the slow one.

This script runs the server on two pinned CPUs and the client on the rest,
which is the closest a plain workstation gets to the judge's allocation:
the server has two CPUs of compute, the client is not competing for them,
and the server still sees the whole machine in its affinity mask.

Usage:
    python bench_judge.py [--rows 10000] [--queries 1000] [--cpus 0,1]
                           [--both-pinned] [--data-dir DIR]

By default the server is not pinned: it chooses the fast cores itself, the
way it will on the judge.  Passing --cpus pins it with taskset, which also
narrows the set of cores the program may pick from -- so pin it onto fast
cores, or not at all.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.dirname(HERE)
MAIN = os.path.join(WS, "main.py")
PYTHON = sys.executable


def vector_literal(values):
    return "[" + ",".join("%.6f" % value for value in values) + "]"


def build_dataset(rows, dim, categories, rng):
    data = []
    for index in range(rows):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        data.append((index + 1, "cat-%d" % (index % categories),
                     "payload-%d" % index, vector))
    return data


class Db:
    def __init__(self, data_dir, log_path, cpus=None):
        command = []
        if cpus:
            command += ["taskset", "-c", cpus]
        command += [PYTHON, "-u", MAIN]
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(log_path, "ab", buffering=0),
            env=dict(os.environ, VECTORDB_DATA_DIR=data_dir))
        self.seq = 0
        ready = self._readline()
        if ready != {"type": "ready"}:
            raise RuntimeError("expected ready, got %r" % (ready,))

    def _readline(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("database exited before response")
        return json.loads(line.decode("utf-8"))

    def exec(self, sql):
        self.seq += 1
        request = {"id": self.seq, "op": "execute", "sql": sql, "search_mode": "exact"}
        self.proc.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        self.proc.stdin.flush()
        response = self._readline()
        if response.get("id") != request["id"]:
            raise RuntimeError("response id differs: %r" % (response,))
        return response

    def shutdown(self):
        self.seq += 1
        self.proc.stdin.write((json.dumps({"id": self.seq, "op": "shutdown"}) + "\n").encode())
        self.proc.stdin.flush()
        self.proc.stdout.readline()
        self.proc.stdin.close()
        self.proc.wait(timeout=60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    # Empty by default: the server picks its own cores, and it picks the
    # fast ones (minivectordb/_pick_cores reads cpu_capacity).  Pinning it
    # with taskset first narrows the mask the program can choose from, so a
    # pin onto cores 0 and 1 -- which on a heterogeneous machine are the
    # slow ones -- measures the small cores and nothing else.  That measured
    # 0.29 ms/query against 0.10 ms/query for the same code, a factor of
    # three, all of it the pin and none of it the engine.
    parser.add_argument("--cpus", default="",
                        help="taskset mask for the server; empty lets it pick "
                             "the fast cores itself (the default)")
    parser.add_argument("--both-pinned", action="store_true")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args(argv)

    root = args.data_dir or os.path.join(
        "/home/wzh_linux/WorkBuddy/2026-11-14-06-20-42/tmp/judge", args.tag)
    shutil.rmtree(root, ignore_errors=True)
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    rng = random.Random(args.seed)
    data = build_dataset(args.rows, args.dim, args.categories, rng)
    queries = []
    for index in range(args.queries):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
        category = "cat-%d" % (index % args.categories) if index % 2 == 0 else None
        queries.append((vector, category))

    server_cpus = args.cpus
    client_cpus = args.cpus if args.both_pinned else None
    db = Db(data_dir, os.path.join(root, "stderr.log"), server_cpus)
    client_command = []
    if client_cpus:
        client_command += ["taskset", "-c", client_cpus]
    try:
        db.exec("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                "content VARCHAR(64), embedding VECTOR(%d));" % args.dim)
        write_start = time.perf_counter()
        for offset in range(0, len(data), 100):
            chunk = data[offset:offset + 100]
            values = ",".join("(%d,'%s','%s','%s')" % (pk, category, content, vector_literal(vector))
                              for pk, category, content, vector in chunk)
            response = db.exec("INSERT INTO docs VALUES %s;" % values)
            if not response.get("ok"):
                raise RuntimeError("insert failed: %r" % (response,))
        write_seconds = time.perf_counter() - write_start
        db.exec("CREATE VECTOR INDEX docs_vec ON docs(embedding);")

        statements = []
        for vector, category in queries:
            if category is None:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                       "ORDER BY distance LIMIT %d;" % (vector_literal(vector), args.k))
            else:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                       "WHERE category = '%s' ORDER BY distance LIMIT %d;"
                       % (vector_literal(vector), category, args.k))
            statements.append(sql)

        # one warm up round, then the measured one
        for sql in statements[:50]:
            db.exec(sql)
        start = time.perf_counter()
        for sql in statements:
            response = db.exec(sql)
            if not response.get("ok"):
                raise RuntimeError("query failed: %r" % (response,))
        elapsed = time.perf_counter() - start
        per_query = elapsed / len(statements) * 1000.0

        # The protocol floor: the same pipe, the same client, a statement
        # that answers without touching the vector column.  Whatever it costs
        # is what every query pays before the engine starts.
        trivial = "SELECT 1;" * 1
        for _ in range(50):
            db.exec(trivial)
        floor_start = time.perf_counter()
        for _ in range(len(statements)):
            db.exec(trivial)
        floor = (time.perf_counter() - floor_start) / len(statements) * 1000.0

        # The score formula of the task: E = (3.67/w + 2 * 664.335/q) / 3,
        # with q in milliseconds.
        score = (3.67 / max(write_seconds, 1e-9) + 2 * 664.335 / per_query) / 3.0
        print("server cpus=%s  client cpus=%s" % (server_cpus, client_cpus or "all"))
        print("write %.3f s   query %.4f ms/query   protocol floor %.4f ms   "
              "engine %.4f ms   E estimate %.0f"
              % (write_seconds, per_query, floor, per_query - floor, score))
    finally:
        try:
            db.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
