# MiniVectorDB —— 提交给判题器的镜像
#
# 判题器执行: docker run -i --network=none --cpus=2 --memory=2g --memory-swap=2g \
#                 --pids-limit=128 --read-only --cap-drop=ALL --security-opt=no-new-privileges \
#                 --mount type=bind,src=<data>,dst=/data <image>
# 因此镜像入口必须直接进入 stdin/stdout 的 JSONL 模式，不能开端口、不能进交互终端。
#
# 依赖策略: 正式执行时禁止联网，依赖必须在构建阶段安装完成并锁定版本。
# numpy 是通用数学库（赛题允许使用「通用文件、序列化、数学及 SQL 解析库」），是查询项性能的
# 关键杠杆（向量 TopK 查询占性能权重 2/3）。
#
# 轮子随仓库一起提交（vendor/），构建时**先装本地轮子**，不依赖构建环境的网络：
# 判题正式执行时是 --network=none，如果构建期 pip 装不上 numpy，程序会退回纯标准库路径，
# TopK 查询从 ~5 ms 变成 ~1250 ms，1000 条查询要 20 分钟，既超时也拿不到性能分。
# 三个安装分支依次是：本地轮子 -> 网络 -> 放弃（纯标准库，功能仍正确）。
FROM python:3.12-slim

# 判题器给的是 --cpus=2，那是 CFS 带宽配额，不是 cpuset：容器在亲和性掩码里仍然
# 看得到宿主机全部核，按掩码开线程池的线性代数库会开出一堆跑不动的 worker。
# 每一条 TopK 查询就是一次 10000x128 的矩阵向量积，本机实测：2 个 worker 38 us，
# 1 个 worker 69 us，按掩码放开则是 14 us / 413 us 的双峰——均值被拖到 413 us，
# 而分数用的正是均值。判题器只给两个 CPU，所以池子就是两个。
# 这里与 minivectordb/__init__.py 的 setdefault 是同一件事，
# 运维仍可用 MINIVECTORDB_BLAS_THREADS 覆盖。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VECTORDB_DATA_DIR=/data \
    OPENBLAS_NUM_THREADS=2 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    MINIVECTORDB_BLAS_THREADS=2

WORKDIR /app

# numpy 2.2.3 的 cp312 manylinux_2_17_x86_64 轮子（16 MB）。
# --only-binary=:all: 保证 pip 不会退化成源码编译（镜像里没有编译器，源码编译必然失败）。
COPY vendor /app/vendor
RUN pip install --no-cache-dir --disable-pip-version-check --only-binary=:all: /app/vendor/*.whl \
    || pip install --no-cache-dir --disable-pip-version-check --only-binary=:all: "numpy==2.2.3" \
    || echo "numpy unavailable, falling back to the pure standard library path"

COPY main.py /app/main.py
COPY minivectordb /app/minivectordb

# /data 由判题器以 bind mount 方式挂载并可写；根文件系统是只读的，
# 因此程序运行时不会向 /app 或其它根目录路径写入任何文件。
VOLUME ["/data"]
CMD ["python3", "-u", "/app/main.py"]
