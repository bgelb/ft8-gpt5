#!/usr/bin/env bash
set -euo pipefail
DATA_DIR=${1:-data/ft8_wav}
COMMIT=${2:-50ee0c06361388a992c80a1af9c1189652b72e51}
mkdir -p "$DATA_DIR"
TMP_DIR=$(mktemp -d)
trap "rm -rf $TMP_DIR" EXIT
curl -sL "https://github.com/kgoba/ft8_lib/archive/$COMMIT.zip" -o "$TMP_DIR/ft8_lib.zip"
unzip -q "$TMP_DIR/ft8_lib.zip" -d "$TMP_DIR"
rsync -a "$TMP_DIR/ft8_lib-$COMMIT/test/wav/" "$DATA_DIR/"
# Also pull reference decodes (txt files) alongside wavs
