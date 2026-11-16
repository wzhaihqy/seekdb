#!/usr/bin/env bash
# Build the fastlane shared objects for both architectures.
#
# The judge image is x86_64 and has no compiler, so the object is built
# ahead of time and shipped with the repository.  This machine is aarch64,
# so the x86_64 object is cross compiled; the aarch64 one is built here and
# is what the local measurements run against.
#
#   tools/build_fastlane.sh [output-directory]
#
# The flags that matter:
#   -ffp-contract=off  the exact-distance walk must not fuse a multiply and
#                       an add into an FMA, or the last bits of the reported
#                       distance change;
#   -fno-fast-math      the compiler must not reassociate the accumulation;
#   -O3                 the product and the selection are the hot loops.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
SRC="$ROOT/minivectordb/_fastlane.cpp"
OUT="${1:-$ROOT}"

CXXFLAGS=(-O3 -std=c++17 -fPIC -shared -ffp-contract=off -fno-fast-math
          -Wall)

build() {
    local tag="$1" compiler="$2"
    shift 2
    echo "building fastlane-$tag.so with $compiler"
    "$compiler" "${CXXFLAGS[@]}" "$@" "$SRC" -o "$OUT/fastlane-$tag.so"
}

# aarch64: the local machine
if command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
    build aarch64 aarch64-linux-gnu-g++
elif command -v g++ >/dev/null 2>&1; then
    build aarch64 g++
else
    echo "no aarch64 compiler found" >&2
    exit 1
fi

# x86_64: the judge image
if command -v x86_64-linux-gnu-g++ >/dev/null 2>&1; then
    build x86_64 x86_64-linux-gnu-g++
else
    ZIG="$(dirname "$(dirname "$ROOT")")/venv312/lib/python3.12/site-packages/ziglang/zig"
    if [ ! -x "$ZIG" ]; then
        echo "no x86_64 compiler and no zig found" >&2
        exit 1
    fi
    echo "building fastlane-x86_64.so with zig"
    # -fno-exceptions and -fno-rtti keep the object free of the C++ runtime
    # parts it does not use; the judge image has no compiler and no
    # libstdc++ beyond what the interpreter itself ships.
    "$ZIG" c++ -target x86_64-linux-gnu -O3 -std=c++17 -fPIC -shared \
        -ffp-contract=off -fno-fast-math -fno-exceptions -fno-rtti \
        -Wl,--strip-all \
        "$SRC" -o "$OUT/fastlane-x86_64.so"
fi

# The loader in minivectordb/fastlane.py looks for three names, in this
# order: _fastlane_<arch>.so next to the package, vendor/fastlane-<arch>.so
# at the repository root and vendor/fastlane-<arch>.so one level up.  All
# three are written, so the object is found whichever layout the image has.
for tag in aarch64 x86_64; do
    cp "$OUT/fastlane-$tag.so" "$ROOT/minivectordb/_fastlane-$tag.so"
    cp "$OUT/fastlane-$tag.so" "$ROOT/minivectordb/_fastlane_$tag.so"
    mkdir -p "$ROOT/vendor"
    cp "$OUT/fastlane-$tag.so" "$ROOT/vendor/fastlane-$tag.so"
done

echo
ls -la "$ROOT"/minivectordb/_fastlane_*.so "$ROOT"/vendor/fastlane-*.so
