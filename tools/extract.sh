#!/bin/bash
# Unpacks a decrypted PS4 package (your own dump) with pkg_pfs_tool.
# Usage: tools/extract.sh <pkg> [out dir]   (default out dir: extracted)
# GT7: the game data is 33 gtNN.vol files, about 88 GB: unpack to a disk with
# room and make sure they end up in Image0/contents next to eboot.bin.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG_TOOL="$DIR/tools/pkg_pfs_tool"
PASSCODE="00000000000000000000000000000000"

PKG_FILE="${1:?usage: tools/extract.sh <pkg> [out dir]}"
OUT_DIR="${2:-"$DIR/extracted"}"

if [ ! -f "$PKG_FILE" ]; then
    echo "PKG file not found: $PKG_FILE" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "Unpacking $PKG_FILE into $OUT_DIR"
# Inner PFS (eboot.bin, sce_module, game data) and the SC entries (param.sfo, icon0.png...).
"$PKG_TOOL" --no-signature-check --no-icv-check --passcode "$PASSCODE" --unpack-sc-entries -u "$PKG_FILE" "$OUT_DIR"
echo "Done"
