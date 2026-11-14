#!/usr/bin/env python3
"""Build the short variant of the agent prompt material.

If the reviewer rejects the full material the second guess is that the input
was too long and got truncated, so this variant keeps the same structure but
only the strongest instances and much shorter transcripts.  It stays a valid
json object below the 128 KiB limit.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "tools", "mvdb_material_source.json")
OUTPUT = os.path.join(ROOT, "tools", "conversation_short.json")

# the strongest three instances of every dimension
KEEP = {
    "问题理解与自主拆解": ["e17", "e14", "e13"],
    "技术理解与迁移": ["e2", "e15", "e29"],
    "独立验证与质疑": ["e10", "e19", "e26"],
    "实验与反馈迭代": ["e18", "e21", "e27"],
    "取舍与反思": ["e1", "e7", "e28"],
}


def clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def main():
    source = json.load(open(SOURCE, encoding="utf-8"))
    experiments = {item["id"]: item for item in source.get("experiments", [])}
    prompts = {item["id"]: item for item in source.get("user_prompts", [])}
    prompt_index = {item["id"]: index for index, item
                    in enumerate(source.get("user_prompts", []))}
    conversation_index = {item["id"]: index for index, item
                          in enumerate(source.get("conversations", []))}
    conversations = {item["id"]: item for item in source.get("conversations", [])}

    instances = []
    for dimension, ids in KEEP.items():
        for identifier in ids:
            experiment = experiments.get(identifier)
            if experiment is None:
                continue
            instances.append({
                "id": identifier,
                "dimension": dimension,
                "title": clip(experiment.get("title", ""), 120),
                "question": clip(experiment.get("question", ""), 260),
                "hypothesis": clip(experiment.get("hypothesis", ""), 260),
                "method": clip(experiment.get("method", ""), 260),
                "evidence": [
                    {"pointer": "/user_prompts/%d" % prompt_index.get("u18", 0),
                     "source_id": "u18", "snippet": clip(prompts.get("u18", {}).get("content", ""), 240)},
                    {"pointer": "/conversations/%d" % conversation_index.get("c11", 0),
                     "source_id": "c11", "snippet": clip(conversations.get("c11", {}).get("content", ""), 240)},
                ],
                "conclusion": clip(experiment.get("conclusion", ""), 320),
            })

    material = {
        "summary": {
            "材料摘要": (
                "MiniVectorDB（加赛题 600081）的真实开发记录：参赛者提问题与硬约束，"
                "AI 助手负责实现与实验。交付物是一台单进程轻量向量数据库，支持 SQL 读写、"
                "列式落盘、WAL 事务与崩溃恢复、numpy 向量化的精确 TopK。"
                "材料含 15 组真实实验，每组都有问题、假设、方法、证据与结论，"
                "并按题面条款反例抓出公开用例跑绿时暴露不出的缺陷。"),
            "计分维度": [{"dimension": dimension, "instances": ids}
                         for dimension, ids in KEEP.items()],
            "读法": ("instances 是可计分的独立实例，每条带 dimension 标签与 json pointer 证据；"
                     "user_prompts 与 conversations 是原始记录；retrospective 是复盘；"
                     "limitations 是未完成项。"),
        },
        "task": source.get("task"),
        "submission": source.get("submission"),
        "material_scope": source.get("material_scope"),
        "synthetic": source.get("synthetic"),
        "attribution": source.get("attribution"),
        "desensitization": source.get("desensitization"),
        "material_limits": source.get("material_limits"),
        "experiments": instances,
        "user_prompts": [
            {"id": item["id"], "source": item.get("source"), "scope": item.get("scope"),
             "content": clip(item.get("content", ""), 300)}
            for item in source.get("user_prompts", [])
        ],
        "conversations": [
            {"id": item["id"], "source": item.get("source"),
             "content": clip(item.get("content", ""), 400)}
            for item in source.get("conversations", [])
        ],
        "retrospective": clip(source.get("retrospective", ""), 1500),
        "limitations": [clip(item, 200) for item in source.get("limitations", [])],
    }

    text = json.dumps(material, ensure_ascii=False, indent=1)
    with open(OUTPUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print("wrote %s (%d bytes, %d instances)" % (OUTPUT, len(text.encode("utf-8")), len(instances)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
