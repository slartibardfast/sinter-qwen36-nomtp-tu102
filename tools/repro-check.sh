#!/usr/bin/env bash
# Double-build byte-identity check (the reproducible-build discipline the
# host's .host-software records). Builds the tree twice from scratch into two
# throwaway dirs and compares artifact hashes; exits nonzero on any mismatch.
set -euo pipefail
SRC="${1:?usage: repro-check.sh <source-dir>}"

hash_build() {
    local dir="$1"
    rm -rf "$dir"
    cmake -S "$SRC" -B "$dir" -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build "$dir" -j > /dev/null
    (cd "$dir" && find . -maxdepth 2 -type f \( -perm -111 -o -name '*.so' \) \
        ! -path './CMakeFiles/*' -print0 | sort -z | xargs -0 sha256sum)
}

A=$(hash_build /tmp/mk-repro-a)
B=$(hash_build /tmp/mk-repro-b)
if [ "$A" != "$B" ]; then
    echo "repro-check: MISMATCH" >&2
    diff <(echo "$A") <(echo "$B") >&2 || true
    exit 1
fi
echo "repro-check: byte-identical"
echo "$A"
