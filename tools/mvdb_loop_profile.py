#!/usr/bin/env python3
"""Attribute the per query cost of the JSONL server, layer by layer.

``mvdb_profile_e2e.py`` measures the engine in process, where the whole
matrix stays in the caches of the process that just wrote it.  The judge
measures the server behind a pipe, where the matrix is read from a memory
mapping and the client runs on the same machine.  The two numbers differ by
more than the protocol can explain, so this script runs the *server loop
itself* -- the same reads, the same JSON, the same writes -- with a timer
around every step.

The child process reports the four phases on stderr when it is done:

    read      the blocking read of the next request line
    decode    json.loads of the request
    execute   engine.execute
    emit      json.dumps plus the write and the flush

Every report carries the label the parent gave it, because the number of
reports depends on how many times the parent asked for one -- a report
matched to a phase by its position in the log is a report matched to the
wrong phase the moment the order changes.

Usage:
    python mvdb_loop_profile.py [--rows 10000] [--queries 1000]
                                [--dim 128] [--k 10] [--categories 8]
                                [--server-cpus 15,19] [--client-cpus 0-9]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(REPO, "main.py")

CHILD = r'''
import json, os, sys, time
sys.path.insert(0, __REPO__)
from minivectordb.engine import Engine

data_dir = sys.argv[1]
engine = Engine(data_dir)
engine.open()

out = sys.stdout
out.write(json.dumps({"type": "ready"}) + "\n")
out.flush()

timings = {"read": 0.0, "decode": 0.0, "execute": 0.0, "emit": 0.0}
count = 0
stdin = sys.stdin.buffer
while True:
    t0 = time.perf_counter()
    raw = stdin.readline()
    if not raw:
        break
    t1 = time.perf_counter()
    line = raw.strip()
    if not line:
        continue
    request = json.loads(line.decode("utf-8"))
    t2 = time.perf_counter()
    if request.get("op") == "shutdown":
        out.write(json.dumps({"id": request.get("id"), "ok": True}) + "\n")
        out.flush()
        break
    if request.get("op") == "mark":
        # Report what has been measured so far and start over, so that the
        # report covers the query phase only and not the load that precedes
        # it, and so that a second phase can be measured the same way.
        # The label travels with the report: the parent knows which phase
        # it asked about, and a report read back out of a log has to be
        # able to say so itself.
        sys.stderr.write("LOOP " + json.dumps(timings) + " count=%d label=%s\n"
                         % (count, request.get("label", "")))
        sys.stderr.flush()
        timings = {"read": 0.0, "decode": 0.0, "execute": 0.0, "emit": 0.0}
        count = 0
        out.write(json.dumps({"id": request.get("id"), "ok": True}) + "\n")
        out.flush()
        continue
    try:
        result = engine.execute(request.get("sql", ""), request.get("search_mode", "exact"))
    except Exception as exc:
        result = {"error": {"code": "internal_error", "message": str(exc)}}
    response = {"id": request.get("id"), "ok": True}
    response.update(result)
    t3 = time.perf_counter()
    out.write(json.dumps(response, ensure_ascii=True, allow_nan=False,
                         separators=(",", ":")) + "\n")
    out.flush()
    t4 = time.perf_counter()
    timings["read"] += t1 - t0
    timings["decode"] += t2 - t1
    timings["execute"] += t3 - t2
    timings["emit"] += t4 - t3
    count += 1

sys.stderr.write("LOOP " + json.dumps(timings) + " count=%d label=requests pre-encoded\n"
                 % count)
engine.close()
'''


def vector_literal(values):
    return "[" + ",".join("%.6f" % value for value in values) + "]"


def _parse_cpus(spec):
    """A cpu list such as ``0-3,8`` as a set of core numbers."""

    cores = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, last = part.split("-", 1)
            cores.update(range(int(first), int(last) + 1))
        else:
            cores.add(int(part))
    return cores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--client-cpus", default="",
                        help="taskset mask for the client; empty lets it run "
                             "anywhere, which lets it compete with the server "
                             "for the cores the server picked")
    args = parser.parse_args(argv)

    # The data directory holds the whole column -- ten thousand rows of a
    # hundred and twenty eight float32 values, plus the write ahead log --
    # which is several megabytes.  ``/tmp`` is a ten megabyte tmpfs on this
    # machine, so a temporary directory there fills up during the load and
    # the write fails with ENOSPC.  The scratch space lives next to the
    # repository instead, on the disk that has room for it.
    scratch = os.environ.get("MVDB_SCRATCH") or os.path.join(
        os.path.dirname(REPO), "tmp", "loop-profile")
    os.makedirs(scratch, exist_ok=True)
    root = tempfile.mkdtemp(prefix="mvdb-loop-", dir=scratch)
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)

    rng = random.Random(args.seed)
    data = []
    for pk in range(1, args.rows + 1):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
        data.append((pk, "cat-%d" % (pk % args.categories), "row-%d" % pk, vector))

    child_source = CHILD.replace("__REPO__", repr(REPO))
    child_path = os.path.join(root, "child.py")
    with open(child_path, "w", encoding="utf-8") as handle:
        handle.write(child_source)

    log = open(os.path.join(root, "child.log"), "wb", buffering=0)
    child_command = [sys.executable, "-u", child_path, data_dir]
    # The client is this process and the server is the child.  The server
    # picks its own cores (the fast ones); the client is pushed onto the
    # ones the server did not take, so that the two do not compete for the
    # same core.  On the judge the container has a two core quota and the
    # driver shares it with the program, which is what this reproduces.
    #
    # The affinity is set on the *child*, in a preexec hook: a child
    # inherits the affinity mask of the parent, so setting it on this
    # process before the fork would put the server on the client's cores
    # and leave the two fighting over them.
    preexec = None
    if args.client_cpus:
        client_cores = _parse_cpus(args.client_cpus)
        server_cores = sorted(set(range(os.cpu_count() or 1)) - client_cores)
        if not server_cores:
            server_cores = sorted(client_cores)

        def preexec():                          # noqa: E306 - runs in the child
            os.sched_setaffinity(0, server_cores)

    proc = subprocess.Popen(child_command,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=log, env=dict(os.environ,
                                                 VECTORDB_DATA_DIR=data_dir),
                            preexec_fn=preexec)
    try:
        ready = proc.stdout.readline()
        if not ready:
            raise RuntimeError("child died: %s" % open(log.name).read())

        def send(sql):
            request = {"id": 1, "op": "execute", "sql": sql, "search_mode": "exact"}
            try:
                proc.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode())
                proc.stdin.flush()
            except BrokenPipeError:
                raise RuntimeError("child died: %s" % open(log.name).read())
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("child exited early: %s" % open(log.name).read())
            return json.loads(line.decode())

        def send_prebuilt(payload):
            """Send a request whose bytes are already encoded.

            The judge builds its requests once and then measures how long the
            server takes to answer them.  A client that spends a hundred
            microseconds encoding every request between two responses adds
            that time to the measurement without the server doing anything,
            so the two shapes are measured separately.
            """
            try:
                proc.stdin.write(payload)
                proc.stdin.flush()
            except BrokenPipeError:
                raise RuntimeError("child died: %s" % open(log.name).read())
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("child exited early: %s" % open(log.name).read())
            return json.loads(line.decode())

        send("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
             "content VARCHAR(64), embedding VECTOR(%d));" % args.dim)
        for offset in range(0, len(data), 100):
            chunk = data[offset:offset + 100]
            values = ",".join("(%d,'%s','%s','%s')" % (pk, category, content, vector_literal(vector))
                              for pk, category, content, vector in chunk)
            send("INSERT INTO docs VALUES %s;" % values)
        send("CREATE VECTOR INDEX docs_vec ON docs(embedding);")

        statements = []
        for index in range(args.queries):
            vector = [rng.uniform(-1.0, 1.0) for _ in range(args.dim)]
            category = "cat-%d" % (index % args.categories) if index % 2 == 0 else None
            if category is None:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                       "ORDER BY distance LIMIT %d;" % (vector_literal(vector), args.k))
            else:
                sql = ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                       "WHERE category = '%s' ORDER BY distance LIMIT %d;"
                       % (vector_literal(vector), category, args.k))
            statements.append(sql)

        for sql in statements[:args.warmup]:
            send(sql)

        # Second shape: the request bytes are encoded once, before the
        # measured run, so the client does no work between two responses.
        # The difference between the two runs is what the client costs.
        prebuilt = [(json.dumps({"id": 1, "op": "execute", "sql": sql,
                                 "search_mode": "exact"},
                                ensure_ascii=False) + "\n").encode()
                    for sql in statements]

        # The child reports what it has measured so far when it is asked to,
        # and the label travels with the request: the parent knows which run
        # the report belongs to, so a report is matched to its run by name
        # and not by its position in the log.  The first mark discards the
        # load and the warm up, so that the report that follows covers the
        # query phase only.
        def mark(label):
            payload = {"id": 0, "op": "mark", "label": label}
            proc.stdin.write((json.dumps(payload) + "\n").encode())
            proc.stdin.flush()
            proc.stdout.readline()

        mark("discarded load")
        start = time.perf_counter()
        for sql in statements:
            send(sql)
        wall = (time.perf_counter() - start) / len(statements) * 1000.0

        mark("client encodes each request")
        start = time.perf_counter()
        for payload in prebuilt:
            send_prebuilt(payload)
        wall_prebuilt = (time.perf_counter() - start) / len(statements) * 1000.0
        report_prebuilt = None

        send("SELECT 1;")
        request = {"id": 2, "op": "shutdown"}
        proc.stdin.write((json.dumps(request) + "\n").encode())
        proc.stdin.flush()
        proc.stdout.readline()
        proc.stdin.close()
        proc.wait(timeout=60)
        log.close()

        report = None
        reports = {}
        with open(log.name, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("LOOP "):
                    continue
                payload = line[5:].strip()
                # The report is ``{json} count=N label=NAME``.  The JSON holds
                # spaces of its own, so it is read by brace matching and the
                # fields after it are read from the remainder.
                end = payload.find("}")
                item = json.loads(payload[:end + 1])
                rest = payload[end + 1:].strip()
                # ``count=N`` comes first and is a single token; the label is
                # the rest of the line, and it holds spaces of its own.
                if rest.startswith("count="):
                    rest = rest[len("count="):]
                    head, _, tail = rest.partition(" ")
                    item["count"] = int(head)
                    if tail.startswith("label="):
                        item["label"] = tail[len("label="):]
                else:
                    item["count"] = 0
                label = item.get("label", "")
                # A report is matched to its run by the label the parent sent
                # with the mark, not by its position in the log: the child
                # writes one report per mark, and the number of marks is not
                # the same in every run.
                reports[label] = item
                report = item
        if report is None:
            print(open(log.name).read())
            raise RuntimeError("no report from the child")

        for label in ("client encodes each request", "requests pre-encoded"):
            item = reports.get(label)
            if item is None:
                continue
            n = float(item["count"])
            print("=== server loop: %s ===" % label)
            for phase in ("read", "decode", "execute", "emit"):
                print("  %-8s : %8.1f us" % (phase, item[phase] / n * 1e6))
            server_total = sum(item[p] for p in ("read", "decode", "execute", "emit"))
            print("  %-8s : %8.1f us   (queries=%d)" % ("server", server_total / n * 1e6, item["count"]))
            print()
        print("  wall, client encodes each request : %8.1f us" % (wall * 1000))
        print("  wall, requests pre-encoded         : %8.1f us" % (wall_prebuilt * 1000))
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
