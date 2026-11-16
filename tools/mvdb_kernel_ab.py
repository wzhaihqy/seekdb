"""The kernel on and the kernel off, measured in the same minutes.

Two processes, one with the kernel and one without, each measuring seven
rounds over the same data.  The point is not the absolute number -- the
machine this runs on has other applications on it -- but the difference
between two processes that were alive at the same time and measured the
same thing.  Anything that steals CPU hits both sides, so it cancels.

This is the only measurement that answers the question the platform will
ask: does the kernel get loaded, and what does it buy?  A single-process
comparison cannot answer it, because minivectordb.fastlane decides once
per process whether to load the object, and caches the answer.

Usage:
    python tools/mvdb_kernel_ab.py [--rounds 7] [--queries 1000]
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PYTHON = sys.executable
ROWS = 10000
DIM = 128


def literal(v):
    return "[" + ",".join("%.6f" % x for x in v) + "]"

def measure(disable, rounds, queries, rows, dim):
    global ROWS, DIM
    ROWS = rows
    DIM = dim
    root = os.path.join(REPO, ".mvdb_tmp", "kab", "off" if disable else "on")
    shutil.rmtree(root, ignore_errors=True)
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    env = dict(os.environ, VECTORDB_DATA_DIR=data_dir)
    if disable:
        env["MINIVECTORDB_DISABLE_FASTLANE"] = "1"
    proc = subprocess.Popen([PYTHON, "-u", os.path.join(REPO, "main.py")],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, env=env)
    seq = [0]
    def call(op, **kw):
        seq[0] += 1
        req = {"id": seq[0], "op": op}; req.update(kw)
        proc.stdin.write((json.dumps(req) + "\n").encode()); proc.stdin.flush()
        return json.loads(proc.stdout.readline().decode())
    call("execute", sql="CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                          "content VARCHAR(64), embedding VECTOR(%d));" % DIM)
    rng = random.Random(20261112)
    for off in range(0, ROWS, 100):
        vals = ",".join("(%d,'cat-%d','payload-%d','%s')" % (
            i, i % 8, i, literal([rng.uniform(-1,1) for _ in range(DIM)]))
            for i in range(off+1, min(off+101, ROWS+1)))
        call("execute", sql="INSERT INTO docs VALUES %s;" % vals)
    write_seconds = None
    call("execute", sql="CREATE VECTOR INDEX docs_vec ON docs(embedding);")
    statements = []
    for index in range(queries):
        v = literal([rng.uniform(-1,1) for _ in range(DIM)])
        if index % 2 == 0:
            sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                   "ORDER BY distance LIMIT 10;" % v)
        else:
            sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                   "WHERE category = 'cat-%d' ORDER BY distance LIMIT 10;" % (v, index % 8))
        statements.append(sql)
    for sql in statements[:50]:
        call("execute", sql=sql)
    times = []
    for round_index in range(rounds):
        start = time.perf_counter()
        for sql in statements:
            call("execute", sql=sql)
        times.append((time.perf_counter() - start) / queries * 1000.0)
    call("shutdown"); proc.wait(timeout=60)
    return times

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rounds", type=int, default=7)
parser.add_argument("--queries", type=int, default=1000)
parser.add_argument("--rows", type=int, default=10000)
parser.add_argument("--dim", type=int, default=128)
args = parser.parse_args()

on = measure(False, args.rounds, args.queries, args.rows, args.dim)
off = measure(True, args.rounds, args.queries, args.rows, args.dim)
print("kernel on : " + "  ".join("%.4f" % t for t in on))
print("kernel off: " + "  ".join("%.4f" % t for t in off))
on_min, off_min = min(on), min(off)
on_med = sorted(on)[len(on)//2]; off_med = sorted(off)[len(off)//2]
print()
print("           min      median")
print("on     %8.4f  %8.4f" % (on_min, on_med))
print("off    %8.4f  %8.4f" % (off_min, off_med))
print()
print("kernel speedup (min basis):    %.3fx" % (off_min/on_min))
print("kernel speedup (median basis): %.3fx" % (off_med/on_med))
print("absolute saving (min basis):   %.1f us/query" % ((off_min-on_min)*1000))
