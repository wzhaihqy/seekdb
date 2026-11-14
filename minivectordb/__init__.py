"""MiniVectorDB — a compact AI-native vector database.

Implements the judge protocol of the OceanBase competition task package
(``2026-competition-round2``): JSON Lines over stdin/stdout, ACID storage
under ``/data``, float32 vector semantics and exact/ANN vector search.
"""

import os as _os

# The linear algebra thread pool, decided before numpy is imported.
#
# A TopK query is one matrix-vector product over a column that fits in the last
# level cache.  The library's own pool costs more than it saves when it is
# sized from the machine: on this 20 core host the same 1000 query run takes
# 1.18 ms per query with the pool sized from the affinity mask and 0.34 ms with
# a single thread, and the pool also produces a tail a median over three rounds
# cannot hide -- the same query alternates between 15 us and 3 ms.
#
# Under-sizing is no better.  The product is 10000x128 = two and a half
# million multiply-adds, which is real work for one core: measured on the
# development machine it costs 69 us with one worker and 38 us with two.  The
# judge gives the container two CPUs (``--cpus=2``) and the client that drives
# it runs outside the container, so two workers is what the machine can
# actually run at once -- the pool should be the quota, not the affinity mask,
# and not one either, because then half of the quota is idle while the product
# runs.
#
# The variables are set with ``setdefault``, so an operator who knows the
# target machine can still choose a different number.
def _blas_threads() -> str:
    chosen = _os.environ.get("MINIVECTORDB_BLAS_THREADS")
    if chosen:
        return chosen
    try:
        visible = len(_os.sched_getaffinity(0))
    except (AttributeError, OSError):        # not Linux
        visible = _os.cpu_count() or 1
    return str(max(1, min(2, visible)))


_BLAS_THREADS = _blas_threads()
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_name, _BLAS_THREADS)
del _name

__all__ = ["core", "engine", "server", "sqlparser"]
