# MiniVectorDB 加赛交付包（题号 600081）

这个压缩包是在 linux 上复现与验收本次交付所需的全部内容。

## 目录

| 路径 | 内容 |
| --- | --- |
| `minivectordb/` | 实现主体（core / engine / sqlparser / store / fastpath / server） |
| `main.py` | 程序入口，走 JSONL 协议与判题机通信 |
| `Dockerfile` | 正式评测用的镜像（构建期装 numpy，禁网也能跑） |
| `vendor/` | 预置的 numpy 轮子（构建期离线安装用） |
| `mysqltest/`、`protocol_check/` | 官方公开题包（20 个用例、断言模块与容差） |
| `tools/mvdb_*.py` | 自测工具：官方用例、准入探针、差分对拍、模糊测试、崩溃恢复、内存、性能基准、查询剖析、材料重建 |
| `tools/mvdb_build_material.py` | 重建 Agent Prompt 材料（幂等） |
| `conversation.json`、`agent/conversation.json` | Agent Prompt 学习材料（评审入口读 `agent/conversation.json`） |
| `AGENT_PROMPT.md`、`DESIGN.md`、`README.md` | 材料正文、设计与验证记录、运行说明 |
| `docs/` | 赛题要点与施工清单、正式用例准入修复说明 |
| `.github/workflows/mvdb-perf.yml` | 性能基准的 CI 作业（可在 GitHub Actions 上复现） |
| `minivectordb-bonus.bundle` | 该分支的完整 git 历史（`git clone` 后可恢复） |

## 一、在 linux 上跑起来

```bash
cd minivectordb-bonus
python3 -m venv .venv && . .venv/bin/activate
pip install numpy            # 装不上也不影响功能，会退回纯标准库路径

# 1) 官方公开用例（用官方断言模块判定，20 个用例）
python3 protocol_check/run.py --image <你的镜像名:标签> --output /tmp/public

# 2) 只跑本机进程版的等价验收（不需要 docker）
python3 tools/mvdb_conformance.py
# 期望：checks=263 failures=0
```

`tools/mvdb_conformance.py` 汇总五套检查：官方 20 个公开用例 + 6 个判题场景、66 条准入语句探针、
21 条快路径语义用例、30 条差分语句、120 条随机对拍查询。差分对拍要求快路径与通用执行器
**逐字节一致**，是这次性能改动的安全网。

## 二、性能基准（官方形状）

```bash
python3 tools/mvdb_bench.py --rows 10000 --queries 1000 --json bench.json
```

形状与官方一致：10000 行 × 128 维、1000 次 TopK（K=10）、一半查询带等值过滤、每 100 行提交一次。
官方数据用了未公开的随机种子，所以只有量级可比。

最近一次 Linux（2 核 CI runner）实测：

| 版本 | 查询 | 写入 |
| --- | --- | --- |
| 改造前 | 1.78 ms/次 | 15.2 ms/批 |
| 改造后（只读投影列 + 候选向量一次取 + 去掉逐元素转换） | **1.38 ms/次** | 13.8 ms/批 |

## 三、看查询开销在哪

```bash
python3 tools/mvdb_query_profile.py --rows 10000 --queries 200
python3 tools/mvdb_float32_check.py      # float32 矩阵乘的误差实测（7.3e-6，未采用）
```

## 四、崩溃与内存

```bash
python3 tools/mvdb_crash_test.py         # 强杀后只保留已提交事务
python3 tools/mvdb_mem_probe.py           # 30 万行 × 64 维的峰值内存
```

## 五、重建 Agent Prompt 材料

```bash
python3 tools/mvdb_build_material.py
python3 tools/validate_material.py        # 校验：指针可解析、无空片段、实例只在一个键下
```

## 六、恢复 git 历史

```bash
git clone minivectordb-bonus.bundle repo && cd repo
git checkout minivectordb-submit
```
