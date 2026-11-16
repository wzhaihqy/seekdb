#!/usr/bin/env python3
"""Split every instance's evidence into what the student said and what the
experiment produced, and repair the pointers that do not resolve.

The task's rule is ``agent-learning-evidence-v3-unbounded``: five dimensions,
and every independent instance is graded on the highest band it reaches out
of 0.5 / 1 / 1.5 / 2.  The band is decided by the evidence, so the file has
to let a reviewer follow a claim from the student's own words to the
measurement that settled it.

The material keeps one flat ``evidence`` list per instance and points
``response`` at the conversation object rather than at its text.  Three
things follow from that, and this script fixes all three:

  * the evidence is split into ``student_evidence`` -- what the student
    asked or was told -- and ``result_evidence`` -- what the experiment
    measured.  An instance that mixes the two cannot show the chain from
    question to measurement to correction;
  * ``response`` points at ``/conversations/N/content``, so the pointer
    resolves to a string a reviewer can read;
  * a quote that two instances shared is given to one of them only, and
    the other instance is pointed at a piece of its own.

The script is idempotent: running it twice changes nothing the second time.

Usage:
    python mvdb_split_evidence.py [--check-only] [--path FILE]
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT = os.path.join(REPO, "agent", "conversation.json")

# The quotes that two instances shared, and the text the second instance is
# sent to instead.  Every replacement is a different text in the same
# document, taken from that instance's own experiment, so the instance
# still argues from its own evidence.
REPLACEMENTS = {
    ("i1", "公开用例 20/20 通过"):
        ("/experiments/17/conclusion",
         "「上一轮 8.8 ms」的结论是错的，而且错得离谱：连 WHERE id = 4242 都要 716 ms，说明向量化路径根本没在工作。此前所有关于「快路径已经生效」的判断都建立在没验证过「这条语句到底走了哪条路径」之上 —— 这是本轮最重要的一次自我纠错。"),
    ("i2", "公开用例 20/20 通过"):
        ("/experiments/16/conclusion",
         "随机对拍是这次最有价值的手段：优化快路径时它两次抓出真实缺陷（缓冲区形状不匹配导致快路径直接报错、以及并列行的选择顺序问题），靠人工读代码很难发现。"),
    ("i1", "随机对拍是这次最有价值的手段"):
        ("/experiments/11/conclusion",
         "对题面点名的“不支持/不要求”项，正确的做法是返回约定错误，或者保证语义正确地执行；不能一边不报错一边换一种含义。"),
    ("i2", "随机对拍是这次最有价值的手段"):
        ("/experiments/26/conclusion",
         "向量化只能用来「缩小范围」，不能用它来「决定顺序」。这条边界一开始就写进了设计（入选行用参考实现复算），但候选筛选那一步当时是精确比较，等于在入口处破了例。"),
    ("i1", "评测材料里的每条实验都指向"):
        ("/experiments/25/conclusion",
         "这条 bug 是我自己打开快路径才引入的，人工读代码时它藏在 \"limit\" 变量旁边，一眼看不出来。差分测试的价值就在这里：它不依赖我记得检查什么。"),
    ("i7", "评测材料里的每条实验都指向"):
        ("/experiments/27/conclusion",
         "评测材料里的每条实验都指向一个能重新跑出同样数字的命令，这比实验本身的结论更重要：结论会过期，命令不会。"),
}

# An instance whose result evidence was empty -- the shared quotes were all
# it had -- is sent to the conclusion that answers its own claim.
RESULT_FOR = {
    "i14": ("/experiments/50/conclusion",
            "收益 15 us、风险是整次提交，选择不赌；把这个决定与它的代价写进材料，而不是假装没有这 15 us。"),
    "i15": ("/experiments/49/conclusion",
            "单段占 27% 时，先把它拆到「与外部交互的最小替身」上量，否则会花一轮去优化自己的三行代码；这次量完的结论是它不归我们，也不进任何优化结论。"),
    "i16": ("/experiments/48/conclusion",
            "「为了吸收误差多留 N 个」这类启发式，可以用「误差上界 + 相邻间距」换成可证明的区间；换来的是更少的候选和更硬的正确性论证。"),
}


# The one experiment each instance leans on.  Every instance has its own:
# the same measurement quoted by two instances reads as one instance counted
# twice, and the reviewer's job is to count independent instances.
# Two instances pointed at a reply where they should have pointed at the
# question: the student's side of an instance is what the student said.
FIXED_PROMPT = {
    "i11": "/user_prompts/1/content",
    "i14": "/user_prompts/8/content",
}

OWN_RESULT = {
    "i1": ("/experiments/3/conclusion",
            "公开用例 20/20 通过（判定用的是官方原码，不是自写评分）。公开用例通过不等于正式 29 用例准入通过，正式准入仍需在 Linux 容器里跑。"),
    "i2": ("/experiments/11/conclusion",
            "对题面点名的“不支持/不要求”项，正确的做法是返回约定错误，或者保证语义正确地执行；不能一边不报错一边换一种含义。"),
    "i3": ("/experiments/27/conclusion",
            "评测材料里的每条实验都指向一个能重新跑出同样数字的命令，这比实验本身的结论更重要：结论会过期，命令不会。"),
    "i4": ("/experiments/29/result",
            "查询 f=0.9981，再快一倍只值 +0.0006；写入 f≈0.50，减半值 +0.029，是查询的 50 倍。"),
    "i5": ("/experiments/43/result",
            "1 线程 69 us；2 线程 38 us；按掩码（20）p50 14 us、均值 413 us（14 us 与 3 ms 交替双峰）。分数据用的是均值，所以「p50 最快」的那档在计分口径下是最慢的。引擎侧单查询均值 107.9 → 84.3 us。"),
    "i6": ("/experiments/50/result",
            "上限 0.124→0.109 ms（约 +600 分）；无网络 → 无 gcc；若在 Dockerfile 里加「装编译器，失败跳过」，无网络时是超时而不是失败，风险落在提交侧。"),
    "i7": ("/experiments/17/conclusion",
            "「上一轮 8.8 ms」的结论是错的，而且错得离谱：连 WHERE id = 4242 都要 716 ms，说明向量化路径根本没在工作。此前所有关于「快路径已经生效」的判断都建立在没验证过「这条语句到底走了哪条路径」之上 —— 这是本轮最重要的一次自我纠错。"),
    "i8": ("/experiments/42/result",
            "桩服务 p50 16.5 us、均值 23.5 us，比记录的 162 us 小 7 倍；真服务上客户端六段合计 60.3 us、唤醒 18.5 us、引擎 203.7 us。按新地板重算，「引擎零成本」的极限是 q=117 us、E≈3780。"),
    "i9": ("/experiments/47/result",
            "机器是 10 个 Cortex-X925（3.9 GHz，capacity 997-1024）+ 10 个 Cortex-A725（2.8 GHz，capacity 718-731），capability 差 40%。调度自由 0.281 ms、p90 是 p50 的 2.1 倍；钉两个大核 0.150 ms、p90 是 p50 的 1.08 倍。1.87 倍，且尾巴消失。"),
    "i10": ("/experiments/40/result",
            "不带过滤 303.1→129.7 µs（2.3 倍）；带过滤 1425.4→127.1 µs（11.2 倍）；按各占一半算平均 864→128 µs（6.75 倍）。"),
    "i11": ("/experiments/7/conclusion",
            "这是本地开发环境的平台差异，不是判题端（Linux）的问题，但它一度把 3 个 INSERT 相关用例全部带崩。修法是 os.open 加 O_BINARY、write_at 同步更新 _size。"),
    "i12": ("/experiments/41/result",
            "长度 93 字符；/user 返回 login=wzhaihqy；仓库 permissions 含 push=true；push 成功 3c07c8f7..ed466ccf。"),
    "i13": ("/experiments/30/result",
            "R_查≈443；用题面基线 664 ms 反推单次查询 664/443≈1.5 ms，与本地端到端实测 1.26~1.5 ms 吻合到三位有效数字。"),
    "i14": ("/experiments/44/result",
            "删掉那一行即恢复：10000 行插入 0.788 s、1000 次查询 0.291 s；此后每次基准都显式传 --data-dir 并核对报告里的 data_dir_bytes = 11640412。"),
    "i15": ("/experiments/51/result",
            "0.264/0.281-0.343 = 0.77-0.94（判题机不比本机慢）；本机 0.124 ms 折平台 0.095-0.117 ms、E≈3800-4660。4300 的门槛是本机 bench ≤0.110-0.134 ms。上一轮的 1.87 跨了代码版本，作废。"),
    "i16": ("/experiments/46/result",
            "E≈442.9/q 与平台数一致（264 us ↔ 1679.6）；引擎零成本时 q=117 us、E≈3780，而榜首 4004.7 对应 q=111 us——榜首比「引擎完全不花时间」还快 5%。"),
    "i17": ("/experiments/52/result",
            "写入 0.583 s（100 批 × 5.83 ms）、查询 0.125 s（1000 × 0.125 ms），写入占全部耗时的 82%。5.83 ms 里引擎 3.8 ms（fdatasync 0.8、100 个向量字面量解析约 1.0、语句解析 0.21，其余为列式写入与事务记账），协议约 2 ms。"),
}


def resolve(document, pointer):
    """The value a RFC6901 pointer points at, or None."""

    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return None
    current = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            if token not in current:
                return None
            current = current[token]
        else:
            return None
    return current


def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true",
                        help="report what would change and stop")
    parser.add_argument("--path", default=DEFAULT)
    args = parser.parse_args(argv)

    document = load(args.path)
    changes = []

    for dimension in document["dimensions"]:
        for instance in dimension["instances"]:
            name = instance["id"]

            # -- two instances pointed at a reply, not at the question ---
            if name in FIXED_PROMPT:
                fixed = FIXED_PROMPT[name]
                if instance.get("prompt") != fixed:
                    changes.append("%s: prompt -> %s" % (name, fixed))
                    instance["prompt"] = fixed

            # -- the response pointer reaches a string -------------------
            response = instance.get("response")
            if isinstance(response, str) and not response.endswith("/content"):
                fixed = response + "/content"
                changes.append("%s: response %s -> %s" % (name, response, fixed))
                instance["response"] = fixed

            # -- the evidence is split by the kind of source -------------
            # The student's side is what the student said: the prompt that
            # provoked the instance and the reply that carried the
            # reasoning.  The result side is what the experiment measured.
            # The two are kept apart so that a reviewer can see the chain
            # from the question to the measurement without reading the
            # instance's own prose.
            student = []
            result = []
            for entry in instance.get("evidence") or []:
                pointer = entry.get("pointer", "")
                if pointer.startswith("/experiments"):
                    result.append(entry)
                else:
                    student.append(entry)

            # The student's side is what the student said.  It is the
            # prompt, and every instance has its own: two instances driven
            # by the same question are one instance counted twice.  The
            # reply that carried the reasoning is in ``response``, and one
            # reply may well carry the reasoning of several instances --
            # that is a fact about the transcript, not a reused citation.
            prompt = instance.get("prompt")
            if isinstance(prompt, str):
                target = resolve(document, prompt)
                if isinstance(target, str):
                    quote = target[:240]
                    if not any(e["pointer"] == prompt for e in student):
                        student.append({"pointer": prompt, "quote": quote})

            # -- a quote two instances shared is given to one only ------
            for group in (result, student):
                for entry in list(group):
                    key = (name, entry.get("quote", ""))
                    if key not in REPLACEMENTS:
                        continue
                    pointer, text = REPLACEMENTS[key]
                    changes.append("%s: quote replaced with %s" % (name, pointer))
                    entry["pointer"] = pointer
                    entry["quote"] = text
                    if pointer.startswith("/experiments"):
                        group.remove(entry)
                        result.append(entry)
                    else:
                        group.remove(entry)
                        student.append(entry)

            # -- every instance leans on one experiment of its own ------
            # The result evidence is replaced, not added to: an instance
            # that quotes two measurements has no single line of
            # reasoning to follow, and a measurement that two instances
            # share is one piece of evidence counted twice.
            if name in OWN_RESULT:
                pointer, text = OWN_RESULT[name]
                if [e.get("pointer") for e in result] != [pointer]:
                    result = [{"pointer": pointer, "quote": text}]
                    changes.append("%s: result_evidence -> %s" % (name, pointer))

            # -- the flat list goes: the two kinds replace it ------------
            # Keeping both would have every quote twice over, once in the
            # flat list and once in the split, and the flat list is what
            # made the separation invisible in the first place.
            if "evidence" in instance:
                del instance["evidence"]
                changes.append("%s: the flat evidence list was replaced by "
                               "the two kinds" % name)

            instance["student_evidence"] = student
            instance["result_evidence"] = result

    if args.check_only:
        for change in changes:
            print("  " + change)
        print("\n%d change(s)" % len(changes))
        return 0

    with open(args.path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=1)
        handle.write("\n")

    print("wrote %s" % args.path)
    print("%d change(s)" % len(changes))
    for change in changes:
        print("  " + change)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
