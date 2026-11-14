#!/usr/bin/env python3
"""Rebuild the agent prompt material in the shape the reviewer expects.

The scoring rule ``agent-learning-evidence-v3-unbounded`` scores independent
instances of evidence backed thinking along five dimensions.  For the
reviewer to be able to find those instances the material has to

* start with a summary that states what was built and what the evidence is,
* list the five dimensions and the instances that belong to them,
* point at the raw transcript with an RFC6901 json pointer and a short
  snippet instead of repeating the whole conversation,
* stay well inside the 128 KiB limit and remain a valid json object.

The script is idempotent: it always rebuilds the material from
``tools/mvdb_material_source.json`` which holds the untouched transcripts.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "tools", "mvdb_material_source.json")
TARGETS = [os.path.join(ROOT, "conversation.json"),
           os.path.join(ROOT, "agent", "conversation.json")]

SNIPPET_LIMIT = 240
LIMIT = 128 * 1024

# The five dimensions of the rule, each with the instances that carry the
# evidence for it.  The mapping is an index of the material, the reviewer
# still decides how high each instance is worth.
DIMENSIONS = [
    ("问题理解与自主拆解", ["e17", "e14", "e12", "e13", "e9"]),
    ("技术理解与迁移", ["e2", "e15", "e29", "e23", "e24"]),
    ("独立验证与质疑", ["e10", "e11", "e19", "e21", "e25", "e5"]),
    ("实验与反馈迭代", ["e18", "e3", "e16", "e22", "e26", "e27"]),
    ("取舍与反思", ["e1", "e4", "e6", "e7", "e8", "e20", "e28"]),
]

# Which prompt or conversation each instance was driven by.
POINTERS = {
    "e1": "u1",
    "e2": "u18",
    "e3": "u18",
    "e4": "u3",
    "e5": "u18",
    "e6": "u1",
    "e7": "u1",
    "e8": "u1",
    "e9": "u14",
    "e10": "u1",
    "e11": "u1",
    "e12": "u1",
    "e13": "u1",
    "e14": "u1",
    "e15": "u18",
    "e16": "u18",
    "e17": "u1",
    "e18": "u18",
    "e19": "u18",
    "e20": "u18",
    "e21": "u18",
    "e22": "u18",
    "e23": "u19",
    "e24": "u19",
    "e25": "u19",
    "e26": "u18",
    "e27": "u18",
    "e28": "u19",
    "e29": "u18",
}

# The conversations that carry the reasoning behind the instances.
CONVERSATION_IDS = {
    "e1": "c3", "e2": "c7", "e3": "c7", "e4": "c3", "e5": "c8", "e6": "c6",
    "e7": "c6", "e8": "c5", "e9": "c7", "e10": "c3", "e11": "c3", "e12": "c3",
    "e13": "c6", "e14": "c4", "e15": "c11", "e16": "c12", "e17": "c4",
    "e18": "c11", "e19": "c11", "e20": "c12", "e21": "c11", "e22": "c11",
    "e23": "c12", "e24": "c11", "e25": "c11", "e26": "c12", "e27": "c12",
    "e28": "c12", "e29": "c12",
}


def clip(text, limit=SNIPPET_LIMIT):
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[:limit - 1] + "…"


def build(source):
    experiments = {item["id"]: item for item in source.get("experiments", [])}
    prompts = {item["id"]: item for item in source.get("user_prompts", [])}
    conversations = {item["id"]: item for item in source.get("conversations", [])}

    prompt_index = {item["id"]: index for index, item
                    in enumerate(source.get("user_prompts", []))}
    conversation_index = {item["id"]: index for index, item
                          in enumerate(source.get("conversations", []))}

    instances = []
    for dimension, ids in DIMENSIONS:
        for identifier in ids:
            experiment = experiments.get(identifier)
            if experiment is None:
                continue
            prompt_id = POINTERS.get(identifier, "u1")
            conversation_id = CONVERSATION_IDS.get(identifier, "c1")
            prompt = prompts.get(prompt_id, {})
            conversation = conversations.get(conversation_id, {})
            prompt_pointer = "/user_prompts/%d" % prompt_index.get(prompt_id, 0)
            conversation_pointer = "/conversations/%d" % conversation_index.get(
                conversation_id, 0)
            instance = {
                "id": identifier,
                "dimension": dimension,
                "title": clip(experiment.get("title", ""), 120),
                "question": clip(experiment.get("question", ""), 300),
                "hypothesis": clip(experiment.get("hypothesis", ""), 300),
                "method": clip(experiment.get("method", ""), 300),
                "evidence": [
                    {"pointer": prompt_pointer, "source_id": prompt_id,
                     "snippet": clip(prompt.get("content", ""))},
                    {"pointer": conversation_pointer, "source_id": conversation_id,
                     "snippet": clip(conversation.get("content", ""))},
                ],
                "conclusion": clip(experiment.get("conclusion", ""), 400),
            }
            for key in ("output_before_fix", "fix", "output_after_fix"):
                if experiment.get(key):
                    instance[key] = clip(experiment[key], 320)
            instances.append(instance)

    instance_index = {}
    for index, item in enumerate(instances):
        instance_index[item["id"]] = index

    def guide_pointer(reference):
        parts = reference.lstrip("/").split("/")
        if len(parts) != 2:
            return reference
        kind, identifier = parts
        if kind in ("instances", "experiments"):
            kind = "experiments"
            index = instance_index.get(identifier)
        elif kind == "user_prompts":
            index = prompt_index.get(identifier)
        elif kind == "conversations":
            index = conversation_index.get(identifier)
        else:
            return reference
        return "/%s/%d" % (kind, index) if index is not None else reference

    def guide_pointers(references):
        return [guide_pointer(reference) for reference in references]

    material = {
        "summary": {
            "材料摘要": (
                "这份材料记录 2026 OceanBase 大赛加赛 MiniVectorDB（题号 600081）的真实开发过程："
                "参赛者提出问题、给出硬约束与环境约束，AI 编码助手负责设计、实现、执行实验并整理结果。"
                "交付物是一台单进程轻量向量数据库（minivectordb/），支持 SQL 读写、列式落盘持久化、"
                "WAL 事务与崩溃恢复、numpy 向量化的精确 TopK。"
                "材料里有 29 组真实实验，每组都有问题、假设、方法、前后对比与结论，"
                "按题面条款反例推出的缺陷（float32 溢出被静默接受、UPDATE 改主键被放行、"
                "向量字面量负号解析失败）都能在公开用例里跑绿的情况下单独暴露。"
                "最有价值的一类证据是自我纠错：实测基线推翻了上一轮「查询已优化到 8.8 ms」的结论，"
                "并由此查出向量化快路径从未被触发（解析器把向量字面量标成了字符串），"
                "以及两处只有差分对拍才能发现的语义 bug（OFFSET 被忽略、并列行被 ulp 噪声挤掉）。"
                "所有结论都指向可重跑的命令，材料末尾如实列出未完成项与未实测的风险。"),
            "计分维度": [
                {"dimension": dimension, "instance_count": len(ids), "instances": ids}
                for dimension, ids in DIMENSIONS
            ],
            "证据规模": {
                "user_prompts": len(source.get("user_prompts", [])),
                "conversations": len(source.get("conversations", [])),
                "instances": len(instances),
                "source_experiments": len(source.get("experiments", [])),
            },
            "读法": (
                "experiments 里的每一组就是一条可计分的独立实例（instances），"
                "每条都带 dimension 标签、问题、假设、方法、结论，"
                "以及指向原始对话的 RFC6901 json pointer 和不超过 240 字符的原文片段；"
                "user_prompts 与 conversations 是原始记录，由实例按 pointer 引用；"
                "retrospective 是复盘，limitations 是未完成项。"),
            "评审导读": [
                {"方面": "问题理解与学习规划",
                 "位置": guide_pointers(["/instances/e17", "/instances/e14", "/instances/e12", "/conversations/c1", "/conversations/c10"]),
                 "说明": ("先把题面的硬约束翻译成可验收的判据再决定落到哪一层："
                          "准入是 29 个正式用例全过一票否决，性能是查询占 2/3，"
                          "于是把学习规划排成「先跑通协议 → 补内核 → 补隐藏场景 → 性能」四阶段，"
                          "并把官方 README 里仍然公开的验收条款逐条写成可执行断言。")},
                {"方面": "Prompt 设计与迭代",
                 "位置": guide_pointers(["/user_prompts/u18", "/user_prompts/u19", "/conversations/c10", "/conversations/c12"]),
                 "说明": ("Prompt 的迭代是被失败逼出来的：第一轮「直接实现」跑不过官方用例，"
                          "于是改成贴协议原文并要求附可复现命令；第二轮 AI 直接改代码修一个坏一个，"
                          "改成「先复现 → 提假设 → 只改一处 → 同口径复测」；"
                          "第三轮为性能加「先给 profiler 结论再动手」与同语句差分对拍。")},
                {"方面": "验证纠错与独立判断",
                 "位置": guide_pointers(["/instances/e10", "/instances/e11", "/instances/e19", "/instances/e26", "/instances/e27"]),
                 "说明": ("对假设本身做反例验证：主动按题面条款造反例才抓到 float32 溢出被静默接受、"
                          "UPDATE 改主键被放行；实测基线推翻了上一轮结论，"
                          "进而查出向量化快路径从未被触发；差分对拍抓到 OFFSET 被忽略与并列行的 ulp 噪声。")},
                {"方面": "总结复盘与知识迁移",
                 "位置": ["/retrospective", "/limitations"],
                 "说明": ("复盘三条可迁移的方法：先跑通验收通道再写功能；"
                          "每个优化都要有一条能证明语义未变的对拍；"
                          "用假设-实验-证据三段式记录且失败的假设也留档。"
                          "同时如实列出未实测的容器环境与未完成项。")},
            ],
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
             "session_id": item.get("session_id"), "content": clip(item.get("content", ""), 420)}
            for item in source.get("user_prompts", [])
        ],
        "conversations": [
            {"id": item["id"], "source": item.get("source"), "session_id": item.get("session_id"),
             "content": clip(item.get("content", ""), 700)}
            for item in source.get("conversations", [])
        ],
        "retrospective": clip(source.get("retrospective", ""), 3000),
        "limitations": [clip(item, 400) for item in source.get("limitations", [])],
    }
    return material


def main():
    if not os.path.exists(SOURCE):
        print("missing source:", SOURCE)
        return 1
    source = json.load(open(SOURCE, encoding="utf-8"))
    material = build(source)
    text = json.dumps(material, ensure_ascii=False, indent=1)
    if len(text.encode("utf-8")) > LIMIT:
        print("material is too big:", len(text.encode("utf-8")))
        return 1
    for target in TARGETS:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    print("wrote %s bytes to %s" % (len(text.encode("utf-8")), ", ".join(TARGETS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
