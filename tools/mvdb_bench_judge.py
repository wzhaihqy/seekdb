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
                           [--platform-k 2.676] [--selftest]

By default the server is not pinned: it chooses the fast cores itself, the
way it will on the judge.  Passing --cpus pins it with taskset, which also
narrows the set of cores the program may pick from -- so pin it onto fast
cores, or not at all.

The run prints three numbers and they are not interchangeable:

    E local     what the formula yields with this machine's query latency.
                Optimistic by exactly the factor k, because the judge only
                has two CPUs and this workstation may open a BLAS worker
                per visible core.
    E platform  what the judge prints, once the query latency is scaled by
                k.  This is the number to compare against a platform run.
    E=5000      the local query latency that would be needed to reach a
                platform score of 5000.  Printed next to the protocol floor
                so the gap is visible instead of implied.
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

# ---------------------------------------------------------------------------
# Score arithmetic, as pure functions so it can be unit tested.
#
# The platform computes E = (3.67 / w + 2 * 664.335 / q_plat) / 3, where
# q_plat is the query latency *measured on the judge*, not on this machine.
# The judge runs the container with --cpus=2, so its q is slower than the
# q of a workstation that may open a BLAS worker per visible core.
#
# The ratio between the two is a property of the hardware, not of the code.
# It was calibrated on two independent platform observations of two
# different commits (see docs/性能标准与判定阈值.md section 2.2):
#
#     986ff36d3: platform 1871.704677, q_local ~ 0.0743 ms
#                 -> q_plat ~ 0.2369 ms, k ~ 3.19
#     02fedb9cf: platform 1477.2,     q_local ~ 0.0961 ms
#                 -> q_plat ~ 0.3002 ms, k ~ 3.12
#
# The write term is left uncalibrated on purpose: it is three orders of
# magnitude smaller than the query term (3.67/w ~ 5.7 against
# 2*664.335/q_plat ~ 5600), so the local write time moves the score by
# less than 0.1 and no separate platform write measurement is needed.
# ---------------------------------------------------------------------------

BASELINE_QUERY_MS = 664.335
BASELINE_WRITE_S = 3.67

# The calibration ratio between this machine and the judge.
#
# The judge runs the container with --cpus=2 and the Dockerfile pins the BLAS
# thread count to 2, so the only way to reproduce its shape here is to pin
# two cores and one BLAS thread.  The ratio was measured that way, with the
# two ends of the ratio taken under the very same configuration:
#
#   986ff36d3 scored 1871.704677 on the platform.  Here, in the judge's
#   shape, its best of fifteen alternating runs was 0.0885 ms, and the
#   platform latency that score implies is 0.2368 ms.  The ratio is
#   0.2368 / 0.0885 = 2.676.
#
# An earlier value of 3.15 was in use.  It was wrong: its denominator was
# measured with the thread count unconstrained, so the ratio also carried
# the difference between "every core" and "two cores", which is not a
# property of the machine but of the configuration.  See
# docs/性能标准与判定阈值.md, section 2.2.
DEFAULT_PLATFORM_K = 2.676


def local_score(write_seconds, per_query_ms):
    """E as the formula reads when q is this machine's q.

    Optimistic by exactly the factor k on the query term: this is NOT the
    number the platform prints.
    """
    return (BASELINE_WRITE_S / max(write_seconds, 1e-9)
            + 2 * BASELINE_QUERY_MS / max(per_query_ms, 1e-9)) / 3.0


def platform_score(write_seconds, per_query_ms, k):
    """E as the platform would print it, given the calibration ratio k.

    q_plat = k * q_local; the write term stays local because it is
    negligible (see the comment above).
    """
    q_plat = max(per_query_ms, 1e-9) * k
    return (BASELINE_WRITE_S / max(write_seconds, 1e-9)
            + 2 * BASELINE_QUERY_MS / q_plat) / 3.0


def query_ms_for_score(target, write_seconds, k):
    """The local query latency that would reach E = target on the platform.

    Returns None when the target is unreachable even at q = 0, i.e. when
    the write term alone already exceeds it.
    """
    residual = 3 * target - BASELINE_WRITE_S / max(write_seconds, 1e-9)
    if residual <= 0:
        return None
    q_plat = 2 * BASELINE_QUERY_MS / residual
    return q_plat / k


def _selftest():
    """Check the score functions against the known platform points.

    The tolerance is relative because the write term on the judge was never
    measured separately: it is carried over from this machine.  The query
    term is three orders of magnitude larger, so the write term moves the
    score by well under one point, and 0.5% is the honest bound.

    Both ends of every ratio are taken in the judge's own shape -- two
    cores, one BLAS thread -- because a ratio taken across two shapes
    carries the configuration difference with it.
    """
    def close(got, want):
        assert abs(got - want) / want < 0.005, (got, want)

    # 986ff36d3: local q 0.0885 ms (best of 15, judge shape), local w
    # 0.736 s -> platform 1871.704677
    close(platform_score(0.736, 0.0885, 2.676), 1871.704677)
    # The local score is strictly the optimistic one.
    assert local_score(0.736, 0.0885) > platform_score(0.736, 0.0885, 2.676)
    # Inverting the platform score reproduces the input latency.
    target = platform_score(0.615, 0.0773, DEFAULT_PLATFORM_K)
    back = query_ms_for_score(target, 0.615, DEFAULT_PLATFORM_K)
    assert abs(back - 0.0773) < 1e-6, back
    # A target the write term already covers has no solution: with w = 1 s
    # the write term contributes 3.67 to the numerator, which already
    # exceeds 3 * 1.0, so no query latency can reach E = 1.
    assert query_ms_for_score(1.0, 1.0, DEFAULT_PLATFORM_K) is None
    # A larger k means a slower judge, so a lower platform score.
    assert (platform_score(0.615, 0.0773, DEFAULT_PLATFORM_K)
            > platform_score(0.615, 0.0773, 3.50))
    return True


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
    parser.add_argument("--platform-k", type=float, default=DEFAULT_PLATFORM_K,
                        help="calibration ratio between the judge's query "
                             "latency and this machine's (see the module "
                             "comment; default %.2f)" % DEFAULT_PLATFORM_K)
    parser.add_argument("--selftest", action="store_true",
                        help="run the score arithmetic checks and exit")
    args = parser.parse_args(argv)

    if args.selftest:
        _selftest()
        print("score arithmetic selftest: PASS")
        return 0

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
        # with q in milliseconds.  Two numbers come out of it and they must
        # not be confused: the local one uses this machine's q, the platform
        # one uses the judge's q, which is k times slower.
        e_local = local_score(write_seconds, per_query)
        e_platform = platform_score(write_seconds, per_query, args.platform_k)
        need_5000 = query_ms_for_score(5000.0, write_seconds, args.platform_k)
        print("server cpus=%s  client cpus=%s" % (server_cpus, client_cpus or "all"))
        print("write %.3f s   query %.4f ms/query   protocol floor %.4f ms   "
              "engine %.4f ms"
              % (write_seconds, per_query, floor, per_query - floor))
        print("  E local    %.0f   (this machine's q; optimistic by k)"
              % e_local)
        print("  E platform %.0f   (q_plat = %.4f ms = k %.2f x %.4f ms; "
              "the number the judge prints)"
              % (e_platform, per_query * args.platform_k, args.platform_k, per_query))
        if need_5000 is None:
            print("  E=5000     unreachable: the write term alone already exceeds it")
        else:
            print("  E=5000     needs query %.4f ms/query on this machine "
                  "(protocol floor here is %.4f ms)"
                  % (need_5000, floor))
    finally:
        try:
            db.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
