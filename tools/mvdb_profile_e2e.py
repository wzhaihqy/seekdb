#!/usr/bin/env python3
"""End-to-end cost breakdown of one vector TopK query over the JSONL protocol.

The judge measures wall-clock time of the whole query sequence, so every
layer the request touches is charged to the score.  This tool reproduces the
judge's shape -- one JSON request per line, one JSON response per line,
send one / wait one -- and then breaks the per-query cost down into:

    read+json.loads | sql parse | engine execute | json.dumps+flush

Run it against the engine in-process (fast) and against a real child process
over a pipe (the shape the judge actually uses).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from minivectordb.engine import Engine            # noqa: E402
from minivectordb.sqlparser import parse           # noqa: E402


def vector_literal(values):
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def build_dataset(rows, dim, categories, rng):
    data = []
    for pk in range(1, rows + 1):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        category = "cat-%d" % (pk % categories)
        content = "row-%d" % pk
        data.append((pk, category, content, vector))
    return data


def make_queries(count, dim, categories, rng):
    out = []
    for index in range(count):
        vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        category = "cat-%d" % (index % categories) if index % 2 == 0 else None
        out.append((vector, category))
    return out


def sql_for(vector, category, dim, k):
    literal = vector_literal(vector)
    if category is None:
        return ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
                "ORDER BY distance LIMIT %d;" % (literal, k))
    return ("SELECT id, L2_DISTANCE(embedding, '%s') AS distance FROM docs "
            "WHERE category = '%s' ORDER BY distance LIMIT %d;"
            % (literal, category, k))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args(argv)

    root = tempfile.mkdtemp(prefix="mvdb-prof-")
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir)

    rng = random.Random(args.seed)
    data = build_dataset(args.rows, args.dim, args.categories, rng)
    queries = make_queries(args.queries, args.dim, args.categories, rng)

    engine = Engine(data_dir)
    engine.open()
    engine.execute("CREATE TABLE docs (id BIGINT PRIMARY KEY, category VARCHAR(16), "
                   "content VARCHAR(64), embedding VECTOR(%d));" % args.dim)
    for offset in range(0, len(data), 100):
        chunk = data[offset:offset + 100]
        values = ",".join("(%d,'%s','%s','%s')" % (pk, category, content, vector_literal(vector))
                          for pk, category, content, vector in chunk)
        engine.execute("INSERT INTO docs VALUES %s;" % values)
    engine.execute("CREATE VECTOR INDEX docs_vec ON docs(embedding);")

    statements = [sql_for(v, c, args.dim, args.k) for v, c in queries]

    # ---- layer 1: sql parse only
    for sql in statements[:50]:
        parse(sql)
    t0 = time.perf_counter()
    for sql in statements:
        parse(sql)
    t_parse = (time.perf_counter() - t0) / len(statements) * 1000.0

    # ---- layer 2: engine.execute (parse + execute + result dict)
    for sql in statements[:50]:
        engine.execute(sql, "exact")
    t0 = time.perf_counter()
    for sql in statements:
        engine.execute(sql, "exact")
    t_engine = (time.perf_counter() - t0) / len(statements) * 1000.0

    # ---- layer 3: json.dumps of the response
    responses = [engine.execute(sql, "exact") for sql in statements[:200]]
    t0 = time.perf_counter()
    for response in responses:
        json.dumps(response, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    t_dump = (time.perf_counter() - t0) / len(responses) * 1000.0

    # ---- layer 4: json.loads of the request
    requests = [json.dumps({"id": i, "op": "execute", "sql": s, "search_mode": "exact"},
                            ensure_ascii=False).encode() for i, s in enumerate(statements[:200])]
    t0 = time.perf_counter()
    for raw in requests:
        json.loads(raw.decode("utf-8"))
    t_load = (time.perf_counter() - t0) / len(requests) * 1000.0

    # ---- layer 5: pure-python reference distance cost for one row set
    print("=== 分层剖析（每查询，微秒）===")
    print("  SQL 解析(parse)              : %8.1f us" % (t_parse * 1000))
    print("  引擎 execute(含解析)         : %8.1f us" % (t_engine * 1000))
    print("  -> 纯执行部分                : %8.1f us" % ((t_engine - t_parse) * 1000))
    print("  json.loads(请求)             : %8.1f us" % (t_load * 1000))
    print("  json.dumps(响应)             : %8.1f us" % (t_dump * 1000))
    print("  端到端预估(层和)             : %8.1f us" % ((t_load + t_engine + t_dump) * 1000))
    print()
    print("请求字节数: 平均 %.0f B  (SQL 里 128 维字面量占大头)" %
          (sum(len(r) for r in requests) / len(requests)))
    print("响应字节数: 平均 %.0f B" %
          (sum(len(json.dumps(r, ensure_ascii=True, allow_nan=False,
                              separators=(",", ":")).encode()) for r in responses) / len(responses)))
    print()
    print("E 估算（用 q = 引擎执行 + 协议开销）:")
    for label, q_ms in (("仅引擎", t_engine),
                        ("引擎+解析+序列化", t_load + t_engine + t_dump)):
        # write time measured separately below
        pass
    engine.close()
    shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
