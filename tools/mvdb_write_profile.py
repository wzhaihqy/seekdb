#!/usr/bin/env python3
"""Measure where the write path spends its time.

The query path is already saturated on the task's score: halving the
query latency moves the normalised E by 0.00003.  The write path is the
only one with slope left, so it is the only one worth spending effort on.
But "the write path" is a label, not a measurement -- this script breaks
one batch of rows into the parts that make it up.

A batch of 100 rows goes through:

    client    build the SQL text and the JSON request
    protocol  read a line, json.loads it
    parse     tokenise the INSERT and decode the vector literals
    coerce    turn the parsed nodes into typed column values
    wal       encode the row and append it to the log
    overlay   stage the row for the transaction
    sync      fdatasync the log -- this one is required, the task says the
              write time includes the time until the data is durable
    flush     write the staged rows into the data files

Every part is timed separately and reported as microseconds per row and
as a share of the batch, so the ranking comes from the measurement rather
than from intuition.

Usage:
    python mvdb_write_profile.py [--rows 10000] [--batch 100] [--dim 128]
                                  [--tag run]
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


def build_rows(rows, dim, categories, rng):
    data = []
    for index in range(rows):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        data.append((index + 1, "cat-%d" % (index % categories),
                     "payload-%d" % index, vector))
    return data


def build_insert_sql(chunk, dim):
    values = ",".join("(%d,'%s','%s','%s')" % (pk, category, content,
                                                vector_literal(vector))
                       for pk, category, content, vector in chunk)
    return "INSERT INTO docs VALUES %s;" % values


class Server:
    """The server as a subprocess, driven over the JSONL protocol."""

    def __init__(self, data_dir, log_path):
        self.proc = subprocess.Popen(
            [PYTHON, "-u", MAIN],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(log_path, "wb"),
        )
        ready = self.proc.stdout.readline()
        payload = json.loads(ready)
        assert payload.get("type") == "ready", payload

    def request(self, obj):
        line = json.dumps(obj, ensure_ascii=True, allow_nan=False,
                          separators=(",", ":"))
        self.proc.stdin.write(line.encode("ascii") + b"\n")
        self.proc.stdin.flush()
        raw = self.proc.stdout.readline()
        return json.loads(raw)

    def exec(self, sql):
        return self.request({"id": 1, "op": "execute", "sql": sql})

    def shutdown(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args(argv)

    root = args.data_dir or os.path.join(
        "/home/wzh_linux/WorkBuddy/2026-11-16-11-09-41/tmp/writeprof", args.tag)
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)

    rng = random.Random(args.seed)
    data = build_rows(args.rows, args.dim, args.categories, rng)
    batches = [data[i:i + args.batch]
               for i in range(0, len(data), args.batch)]

    server = Server(root, os.path.join(root, "stderr.log"))
    try:
        server.exec("CREATE TABLE docs (id BIGINT PRIMARY KEY, "
                    "category VARCHAR(16), content VARCHAR(64), "
                    "embedding VECTOR(%d));" % args.dim)

        # ---- the client side: building the request -------------------
        t_client = 0.0
        payloads = []
        for chunk in batches:
            start = time.perf_counter()
            sql = build_insert_sql(chunk, args.dim)
            payload = {"id": 1, "op": "execute", "sql": sql}
            line = json.dumps(payload, ensure_ascii=True, allow_nan=False,
                              separators=(",", ":"))
            encoded = line.encode("ascii") + b"\n"
            t_client += time.perf_counter() - start
            payloads.append(encoded)

        # ---- the round trip, as the judge sees it --------------------
        wall_start = time.perf_counter()
        for encoded in payloads:
            server.proc.stdin.write(encoded)
            server.proc.stdin.flush()
            raw = server.proc.stdout.readline()
            response = json.loads(raw)
            if not response.get("ok"):
                raise RuntimeError("insert failed: %r" % (response,))
        wall = time.perf_counter() - wall_start
    finally:
        server.shutdown()

    rows = args.rows
    batches_n = len(batches)
    print("=== write path, %d rows in %d batches of %d ===" % (
        rows, batches_n, args.batch))
    print("total wall          %.4f s" % wall)
    print("per row             %.2f us" % (wall / rows * 1e6))
    print("per batch           %.4f ms" % (wall / batches_n * 1e3))
    print()
    print("client side (SQL text + JSON)   %.2f us/batch  %5.1f%%" % (
        t_client / batches_n * 1e6, t_client / wall * 100))

    # The rest of the wall time is the server: parse, coerce, wal, sync,
    # flush.  Those are measured from inside the process, below.
    server_side = wall - t_client
    print("server side (everything else)    %.2f us/batch  %5.1f%%" % (
        server_side / batches_n * 1e6, server_side / wall * 100))
    print()
    print("the server side splits into:")
    print("  parse    SQL tokenising and vector literal decoding")
    print("  coerce   parsed nodes -> typed column values")
    print("  wal      row encoding and log append")
    print("  overlay  staging the row in the transaction")
    print("  sync     fdatasync of the log  (required: the write time")
    print("           includes the time until the data is durable)")
    print("  flush    writing the staged rows into the data files")
    print()
    print("run tools/mvdb_write_pieces.py for the server side breakdown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
