# Tugas PySpark — Big Data & Machine Learning: NYC Taxi Fare Prediction

Analisis data besar dan pemodelan *machine learning* dengan **Apache Spark (PySpark + MLlib)** untuk memprediksi
tarif taksi New York City dari dataset Kaggle
[*New York City Taxi Fare Prediction*](https://www.kaggle.com/competitions/new-york-city-taxi-fare-prediction)
(`train.csv`, **55.423.856 baris, ±5,3 GB**).

| Isi | Lokasi |
|---|---|
| Notebook utama (sudah dieksekusi, lengkap dengan output) | [`notebooks/nyc_taxi_fare_pyspark.ipynb`](notebooks/nyc_taxi_fare_pyspark.ipynb) |
| Laporan (poin 7) | [`laporan/Laporan_Tugas_PySpark.md`](laporan/Laporan_Tugas_PySpark.md) · `.docx` · `.pdf` |
| Grafik hasil EDA & evaluasi | [`figures/`](figures/) |
| Metrik & hasil tuning (JSON/CSV) | [`results/`](results/) |
| Script unduh dataset | [`scripts/download_data.sh`](scripts/download_data.sh) |

## Pemetaan ke poin tugas

| No | Poin tugas | Bobot | Bagian notebook |
|---|---|---|---|
| 1 | Memuat dan menyiapkan data dengan PySpark | 10 | §1 |
| 2 | Eksplorasi data (EDA) | 15 | §2 |
| 3 | Transformasi data / preprocessing | 15 | §3 |
| 4 | Pemodelan dengan PySpark MLlib | 25 | §4 |
| 5 | Evaluasi model dengan metrik yang sesuai | 15 | §5 |
| 6 | Hyperparameter tuning | 10 | §6 |
| 7 | Laporan | 10 | `laporan/` |

## Cara menjalankan

Prasyarat: Python 3.10+ dan Java 17/21 (dibutuhkan Spark).

```bash
pip install -r requirements.txt
bash scripts/download_data.sh            # mirror Parquet (tanpa login), ±3,8 GB ke data/raw/
# atau: bash scripts/download_data.sh kaggle   (file asli train.csv, perlu kaggle.json)

cd notebooks
jupyter notebook nyc_taxi_fare_pyspark.ipynb
```

Notebook membaca `train.csv` (CSV Kaggle) maupun folder Parquet. Variabel lingkungan yang bisa diatur:

| Variabel | Default | Keterangan |
|---|---|---|
| `NYC_TAXI_DATA` | `../data/raw` | file `train.csv` atau folder Parquet |
| `NYC_TAXI_WORK` | `../data/work` | lokasi Parquet hasil olahan |
| `SPARK_MASTER` | `local[*]` | ganti ke URL klaster untuk mode terdistribusi |
| `SPARK_DRIVER_MEMORY` | `10g` | memori driver |
| `MODEL_TRAIN_ROWS` / `TUNING_ROWS` / `FINAL_TRAIN_ROWS` | 2 jt / 500 rb / 10 jt | ukuran data latih tiap tahap pemodelan |
