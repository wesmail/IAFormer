#!/usr/bin/env bash
# Simple JetClass downloader (Zenodo record 6619768)
# Downloads each file sequentially using aria2c

set -e

BASE_URL="https://zenodo.org/records/6619768/files"
DEST_DIR="${1:-JetClass_Zenodo}"
CONN="${ARIA2_CONN:-16}"   # number of parallel connections per file

FILES=(
  JetClass_Pythia_test_20M.tar
  JetClass_Pythia_train_100M_part0.tar
  JetClass_Pythia_train_100M_part1.tar
  JetClass_Pythia_train_100M_part2.tar
  JetClass_Pythia_train_100M_part3.tar
#  JetClass_Pythia_train_100M_part4.tar
#  JetClass_Pythia_train_100M_part5.tar
#  JetClass_Pythia_train_100M_part6.tar
#  JetClass_Pythia_train_100M_part7.tar
#  JetClass_Pythia_train_100M_part8.tar
#  JetClass_Pythia_train_100M_part9.tar
#  JetClass_Pythia_val_5M.tar
)

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

for f in "${FILES[@]}"; do
  echo "[info] Downloading $f ..."
  aria2c -x "$CONN" -s "$CONN" --continue=true "$BASE_URL/$f"
done

echo "[info] All done ✅"

