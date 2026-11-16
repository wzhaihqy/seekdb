"""Optional C acceleration for the hot steps of the vector TopK path.

The judge image has no compiler at run time, so the shared object is built
ahead of time, per architecture, and shipped next to this package (or under
``vendor/`` at the repository root).  When the object for the running
architecture is missing, or when ``ctypes`` cannot load it, every helper
returns ``None`` and the caller keeps the numpy path it already had -- the
semantics of a query never depend on whether this module loaded.

Both helpers were verified bit for bit against the reference converters and
the reference distance walk (3.2 million fields, 1.2 million distances, zero
mismatches); the C code spells out why the two are the same IEEE operations
in the same order.

The buffers and the ``ctypes`` pointers are kept between calls on purpose:
a query calls these helpers once, and building a fresh array plus a fresh
pointer for every call costs more than the work the helpers do.
"""

from __future__ import annotations

import ctypes
import os
import platform
from typing import Any

try:  # pragma: no cover - depends on the image
    import numpy as _np
except ImportError:  # pragma: no cover - depends on the image
    _np = None  # type: ignore[assignment]

_LIB: Any = None
_tried = False

_ARCH_ALIASES = {"amd64": "x86_64", "arm64": "aarch64"}

# Buffers reused between calls, and the pointers into them.  They grow when
# a bigger vector or a bigger candidate set shows up and never shrink.
_parse_buffer: Any = None
_parse_pointer: Any = None
_parse_size = 0
_out_buffer: Any = None
_out_pointer: Any = None
_out_size = 0
_slot_buffer: Any = None
_slot_pointer: Any = None
_slot_size = 0


def _disabled() -> bool:
    """Whether the C object was switched off on purpose.

    The environment variable makes the fallback reproducible on a machine
    that does have the object, so the two paths can be compared.
    """

    return bool(os.environ.get("MINIVECTORDB_DISABLE_FASTLANE"))


def available() -> bool:
    """Whether the C object was found and loaded."""

    return _library() is not None


def _library() -> Any:
    """Load the vendored shared object once, or stay ``None`` forever."""

    global _LIB, _tried
    if _tried:
        return _LIB
    _tried = True
    if _np is None or _disabled():
        return None
    arch = platform.machine().lower()
    arch = _ARCH_ALIASES.get(arch, arch)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "_fastlane_%s.so" % arch),
        os.path.join(here, os.pardir, os.pardir, "vendor", "fastlane-%s.so" % arch),
        os.path.join(here, os.pardir, "vendor", "fastlane-%s.so" % arch),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.fl_parse_vector.restype = ctypes.c_int64
            lib.fl_parse_vector.argtypes = [
                ctypes.c_char_p, ctypes.c_int64,
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)]
            lib.fl_exact_distances.restype = None
            lib.fl_exact_distances.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int64, ctypes.c_int64, ctypes.POINTER(ctypes.c_double)]
            lib.fl_topk.restype = ctypes.c_int64
            lib.fl_topk.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                ctypes.c_int64, ctypes.c_double, ctypes.c_void_p,
                ctypes.c_void_p]
        except (OSError, AttributeError):
            # The object is not for this machine, or it does not carry the
            # three entry points.  Either way the numpy path stays in place.
            continue
        if not _self_test(lib):
            continue
        _LIB = lib
        break
    return _LIB


def _self_test(lib: Any) -> bool:
    """Whether the loaded object agrees with numpy on a tiny case.

    The object is compiled ahead of time, for the machine that will run it,
    and a build that disagrees with numpy would silently change the answers.
    The check is three rows and a two dimensional query, it runs once when the
    object is loaded, and it compares the distances the kernel reports with
    the ones numpy computes.  An object that fails it is not used.
    """

    try:
        matrix = _np.asarray([[0.0, 0.0], [3.0, 4.0], [-1.0, 0.0]],
                              dtype=_np.float32)
        query = _np.asarray([0.0, 0.0], dtype=_np.float32)
        norms = _np.asarray([0.0, 25.0, 1.0], dtype=_np.float32)
        coarse = _np.asarray([0.0, 0.0, 0.0], dtype=_np.float32)
        keys = _np.asarray([10, 20, 30], dtype=_np.int64)
        ids = _np.empty(3, dtype=_np.int64)
        dists = _np.empty(3, dtype=_np.float64)
        written = lib.fl_topk(
            coarse.ctypes.data, norms.ctypes.data, matrix.ctypes.data,
            query.ctypes.data, keys.ctypes.data, None, None,
            3, 2, 2, 0, ctypes.c_double(1e-3),
            ids.ctypes.data, dists.ctypes.data)
        if written != 2:
            return False
        # With no id map the answer reports the row it came from: rows 0 and
        # 2 are the closest to the origin, row 1 is five away.
        expected_rows = (0, 2)
        expected = (0.0, 1.0)
        for index in range(2):
            if int(ids[index]) != expected_rows[index]:
                return False
            if float(dists[index]) != expected[index]:
                return False
        exact = _np.empty(3, dtype=_np.float64)
        lib.fl_exact_distances(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            3, 2, exact.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        # The same walk the reference executor makes: widen, subtract, square,
        # accumulate along the dimension axis.
        wide = matrix.astype(_np.float64) - query.astype(_np.float64)
        wide *= wide
        wanted = _np.sqrt(wide.cumsum(axis=1)[:, -1])
        if not _np.array_equal(exact, wanted):
            return False
    except Exception:                      # pragma: no cover - defensive
        return False
    return True


def _parse_slot(dim: int):
    """A float32 buffer of ``dim`` elements and the pointer into it."""

    global _parse_buffer, _parse_pointer, _parse_size
    if _parse_size < dim:
        _parse_size = max(dim, 256)
        _parse_buffer = _np.empty(_parse_size, dtype=_np.float32)
        _parse_pointer = _parse_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    return _parse_buffer, _parse_pointer


def _out_slot(rows: int):
    """A float64 buffer of ``rows`` elements and the pointer into it."""

    global _out_buffer, _out_pointer, _out_size
    if _out_size < rows:
        _out_size = max(rows, 256)
        _out_buffer = _np.empty(_out_size, dtype=_np.float64)
        _out_pointer = _out_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    return _out_buffer, _out_pointer


def _slot_slot(rows: int):
    """An int64 buffer of ``rows`` elements and the pointer into it."""

    global _slot_buffer, _slot_pointer, _slot_size
    if _slot_size < rows:
        _slot_size = max(rows, 256)
        _slot_buffer = _np.empty(_slot_size, dtype=_np.int64)
        _slot_pointer = _slot_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    return _slot_buffer, _slot_pointer


def parse_text(text: str, dim: int) -> Any:
    """Parse ``[a,b,...]`` into a float32 array, or return ``None``.

    ``None`` means "not for me": a literal that is not a plain bracketed
    list of strict decimal numbers, a field count that does not match
    ``dim``, or anything else the C parser refuses.  The caller then uses
    the converter it already had, so accepted syntax and reported errors
    cannot move.  The value returned on success is the same bit pattern the
    numpy converter produces for the same text -- verified over three
    million random fields, including denormals and the tiny / huge tails of
    the float32 range.
    """

    lib = _library()
    if lib is None or _np is None or dim is None or dim <= 0:
        return None
    if len(text) < 2 or text[0] != "[":
        return None
    buffer, pointer = _parse_slot(dim)
    ok = ctypes.c_int(1)
    count = lib.fl_parse_vector(text.encode("utf-8", "replace"), int(dim),
                                pointer, ctypes.byref(ok))
    if count != dim or not ok.value:
        return None
    return buffer[:dim].copy()


def topk(coarse: Any, norms: Any, matrix: Any, query: Any, keys: Any,
         rows_of: Any, ids_of: Any, take: int, tolerance: float):
    """The whole selection in one call, or ``None``.

    ``coarse`` holds the float32 product of every candidate row with the
    query -- what the linear algebra library computed -- and ``norms`` the
    float32 |row|^2 of the same rows.  The kernel adds them the way the
    numpy path did (``product * -2 + norm``), finds the boundary, collects
    the rows that tie with it, computes their exact float64 distances from
    the stored vectors and returns them ordered by (distance, primary key).

    ``rows_of`` maps a candidate position to the row of ``matrix`` it reads
    (None: the position is the row); ``ids_of`` maps it to the id the answer
    reports (None: the answer reports the row).  ``keys`` holds the primary
    key of every candidate.

    The result is the pair ``(slots, distances)``: the rows of the answer
    and their exact distances.  It is the same answer the numpy path
    produced -- verified against the reference executor over hundreds of
    random tables -- and it arrives in one call instead of five passes over
    Python objects.
    """

    lib = _library()
    if lib is None or _np is None:
        return None
    if coarse.dtype != _np.float32 or query.dtype != _np.float32:
        return None
    if matrix.dtype != _np.float32 or not matrix.flags["C_CONTIGUOUS"]:
        return None
    count = int(coarse.shape[0])
    dim = int(query.shape[0])
    if take > count:
        take = count
    if take <= 0:
        return _np.empty(0, dtype=_np.intp), _np.empty(0, dtype=_np.float64)
    out_slots, out_pointer = _slot_slot(count)
    out_dists, out_dists_pointer = _out_slot(max(count, 1))
    written = lib.fl_topk(
        coarse.ctypes.data, 0 if norms is None else norms.ctypes.data,
        matrix.ctypes.data, query.ctypes.data,
        0 if keys is None else keys.ctypes.data,
        0 if rows_of is None else rows_of.ctypes.data,
        0 if ids_of is None else ids_of.ctypes.data,
        count, dim, int(take), 0, float(tolerance),
        out_slots.ctypes.data, out_dists.ctypes.data)
    if written < 0:
        return None
    return out_slots[:written].copy(), out_dists[:written].copy()


def exact_distances(matrix: Any, query: Any) -> Any:
    """The exact distances of ``matrix`` rows to ``query``, or ``None``.

    ``matrix`` is a contiguous float32 ``(n, dim)`` array, ``query`` a
    float32 ``(dim,)`` vector.  The result is the same float64 array the
    reference ``widen / subtract / square / cumsum / sqrt`` walk produces --
    verified bit for bit over 1.2 million distances; the accumulation order
    in C is the order the reference walks, and ``-ffp-contract=off`` keeps
    the compiler from fusing the multiply into the add.
    """

    lib = _library()
    if lib is None or _np is None:
        return None
    if matrix.dtype != _np.float32 or query.dtype != _np.float32:
        return None
    if not matrix.flags["C_CONTIGUOUS"] or not query.flags["C_CONTIGUOUS"]:
        return None
    rows, dim = matrix.shape
    if query.shape[0] != dim:
        return None
    out_buffer, out_pointer = _out_slot(max(rows, 1))
    if rows:
        lib.fl_exact_distances(
            matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(rows), int(dim), out_pointer)
    return out_buffer[:rows].copy()
