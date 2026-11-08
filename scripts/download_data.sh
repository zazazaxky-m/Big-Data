#!/usr/bin/env bash
# Mengunduh dataset NYC Taxi Fare Prediction (train set, 55.423.856 baris) ke data/raw/
#
# Opsi 1 (default): mirror Parquet di Hugging Face — tanpa login, 12 file ≈ 3,8 GB.
#   bash scripts/download_data.sh
# Opsi 2: file asli dari Kaggle (perlu ~/.kaggle/kaggle.json dan sudah menyetujui aturan kompetisi).
#   bash scripts/download_data.sh kaggle
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/raw"
mkdir -p "${DEST}"

if [[ "${1:-hf}" == "kaggle" ]]; then
  pip install -q kaggle
  kaggle competitions download -c new-york-city-taxi-fare-prediction -f train.csv -p "${DEST}"
  cd "${DEST}" && unzip -o train.csv.zip && rm -f train.csv.zip
  echo "Selesai: ${DEST}/train.csv  (jalankan notebook dengan NYC_TAXI_DATA=../data/raw/train.csv)"
  exit 0
fi

BASE="https://huggingface.co/datasets/shahbajsingh/nyc-taxi-fare-prediction-train/resolve/main/data"
for i in $(seq -w 0 11); do
  f="train-000${i}-of-00012.parquet"
  if [[ -s "${DEST}/${f}" ]]; then echo "lewati ${f} (sudah ada)"; continue; fi
  curl -fL --retry 4 -o "${DEST}/${f}" "${BASE}/${f}" &
done
wait
du -sh "${DEST}"
