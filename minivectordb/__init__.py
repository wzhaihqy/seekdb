"""MiniVectorDB — a compact AI-native vector database.

Implements the judge protocol of the OceanBase competition task package
(``2026-competition-round2``): JSON Lines over stdin/stdout, ACID storage
under ``/data``, float32 vector semantics and exact/ANN vector search.
"""

import os as _os

# One thread for the linear algebra kernel, decided before numpy is imported.
#
# A TopK query is one matrix-vector product over a column that fits in the last
# level cache.  The library's own thread pool costs more than it saves at that
# size: on this 20 core machine the same 1000 query run takes 1.18 ms per query
# with the pool and 0.34 ms per query with one thread, because every call wakes
# the workers and a container throttled by a CPU quota then spends part of its
# budget in the scheduler.  The pool also produces a tail that a median over
# three rounds cannot hide: the same query alternates between 15 us and 3 ms.
#
# The variables are set with ``setdefault``, so an operator who knows the
# target machine can still choose a different number.
_BLAS_THREADS = _os.environ.get("MINIVECTORDB_BLAS_THREADS", "1")
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "BLIS_NUM_THREADS"):
    _os.environ.setdefault(_name, _BLAS_THREADS)
del _name

__all__ = ["core", "engine", "server", "sqlparser"]
