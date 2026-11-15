"""MiniVectorDB — a compact AI-native vector database.

Implements the judge protocol of the OceanBase competition task package
(``2026-competition-round2``): JSON Lines over stdin/stdout, ACID storage
under ``/data``, float32 vector semantics and exact/ANN vector search.
"""

import os as _os

# Where the process runs, and how wide its linear algebra is, decided before
# numpy is imported because the library reads the thread count once.
#
# Two decisions, both measured, both about the same thing: a TopK query is one
# matrix-vector product over a column, and the query stream is one request at
# a time, so the process never needs more than a couple of cores -- it needs
# the *right* couple.
#
# 1. The pool is sized to the CPUs the container can actually run at once.
#    ``--cpus=2`` is a CFS bandwidth quota, not a cpuset: the affinity mask
#    still lists every core of the host, and a BLAS that sizes its pool from
#    the mask starts workers the quota cannot run.  The product is 10000x128,
#    two and a half million multiply-adds: measured on the development machine
#    it costs 77 us with one worker, 40 us with two, and 40 us with four (two
#    physical cores, so four adds nothing but scheduler traffic).
#
# 2. A heterogeneous host is pinned to its fast cores.  The development
#    machine is a 10+10 big.LITTLE part (Cortex-X925 at 3.9 GHz, Cortex-A725
#    at 2.8 GHz); left to the scheduler, the same 1000-query run costs 0.281 ms
#    per query and its p90 is twice its median, because some queries land on a
#    little core and stay there for the whole product.  Pinned to two big
#    cores the same run costs 0.150 ms and p90 is 1.08x the median.  So when
#    the capacity files say the cores are not equal, the process asks for the
#    fast ones; when they are equal (an ordinary VM) nothing is pinned and the
#    scheduler keeps all of its freedom.  Both steps are best effort: a
#    container that forbids ``sched_setaffinity`` is left exactly as it was.
#
# The variables are set with ``setdefault``, so an operator who knows the
# target machine can still choose a different number.
_CAPACITY = "/sys/devices/system/cpu/cpu%d/cpu_capacity"
_SIBLINGS = "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list"


def _physical_core(cpu: int) -> int:
    """The first CPU of this CPU's sibling group: its physical core.

    On an SMT machine two numbered CPUs share one execution unit.  Pinning to
    two *numbers* then pins to one *core*, and two worker threads take turns
    inside a single core instead of running side by side.  Measured on the
    judge host this is what turned a 2.7x local win into a 2.4x loss: the
    capacity files flagged the machine as heterogeneous, the two fastest
    numbers happened to be the two threads of one core, and the pool was
    sized to two.  Siblings are therefore never both taken.
    """

    try:
        with open(_SIBLINGS % cpu, "r", encoding="ascii") as handle:
            spec = handle.read().strip()
    except OSError:
        return cpu                          # no topology: every number is a core
    head = spec.split(",")[0]
    if "-" in head:
        head = head.split("-")[0]
    try:
        return int(head)
    except ValueError:
        return cpu


def _capacity(cpu: int) -> int:
    try:
        with open(_CAPACITY % cpu, "r", encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return 0


def _pick_cores() -> list[int]:
    """The cores this process should run on: the fast ones, if they differ."""

    try:
        allowed = sorted(_os.sched_getaffinity(0))
    except (AttributeError, OSError):        # not Linux: nothing to choose
        return []
    if len(allowed) < 2:
        return []
    capacities = {cpu: _capacity(cpu) for cpu in allowed}
    fastest = max(capacities.values())
    slowest = min(capacities.values())
    if slowest <= 0 or fastest < slowest * 1.25:
        return []                            # uniform cores: leave it alone
    order = sorted(allowed, key=lambda cpu: (-capacities[cpu], cpu))
    wide = [cpu for cpu in order if capacities[cpu] * 4 >= fastest * 3]
    chosen: list[int] = []
    taken: set[int] = set()
    for cpu in wide:
        core = _physical_core(cpu)
        if core in taken:                    # the sibling of a core we already have
            continue
        taken.add(core)
        chosen.append(cpu)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        return []                            # one fast core is worse than two
    return sorted(chosen)


_CHOSEN_CORES = _pick_cores()
if _CHOSEN_CORES:
    try:
        _os.sched_setaffinity(0, set(_CHOSEN_CORES))
    except (AttributeError, OSError):
        _CHOSEN_CORES = []


def _blas_threads() -> str:
    chosen = _os.environ.get("MINIVECTORDB_BLAS_THREADS")
    if chosen:
        return chosen
    if _CHOSEN_CORES:
        return str(len(_CHOSEN_CORES))
    try:
        allowed = _os.sched_getaffinity(0)
    except (AttributeError, OSError):        # not Linux
        return str(max(1, min(2, _os.cpu_count() or 1)))
    # Count physical cores, not numbered CPUs.  A container granted "two CPUs"
    # on an SMT host can own two threads of one core, where a second worker
    # only takes turns with the first; the quota says two, the silicon says one.
    cores = len({_physical_core(cpu) for cpu in allowed})
    return str(max(1, min(2, cores)))


_BLAS_THREADS = _blas_threads()
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_name, _BLAS_THREADS)
del _name

__all__ = ["core", "engine", "server", "sqlparser"]
