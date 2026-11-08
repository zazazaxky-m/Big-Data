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
| Model final (Spark `PipelineModel`) | [`models/gbt_tuned_final/`](models/gbt_tuned_final/) |
| Script unduh dataset | [`scripts/download_data.sh`](scripts/download_data.sh) |

## Hasil utama

Dijalankan pada seluruh 55,4 juta baris (Spark 4.3.0, mode lokal 4 core, ±53 menit untuk seluruh notebook).

| Tahap | Hasil |
|---|---|
| Pembersihan | 2.181.273 baris (3,94%) dibuang → **53.242.583 baris bersih**; split 80/20 → 42,6 jt latih / 10,65 jt uji |
| Temuan EDA | label `UTC` menyesatkan (jam sudah lokal NYC), kenaikan tarif 4 Sep 2012 (+17% per km), tarif flat bandara |
| Fitur | 28 fitur: waktu, jarak haversine/manhattan, arah, jarak ke JFK/LGA/EWR/Midtown |

Evaluasi pada **10.650.052 baris data uji**:

| Model (Spark MLlib) | RMSE (USD) | MAE (USD) | R² |
|---|---:|---:|---:|
| Baseline (rata-rata) | 9,411 | 5,916 | 0,000 |
| Linear Regression | 3,814 | 1,955 | 0,836 |
| Decision Tree | 3,420 | 1,759 | 0,868 |
| Random Forest | 3,308 | 1,732 | 0,876 |
| GBT | 3,170 | 1,573 | 0,887 |
| **GBT hasil tuning (CrossValidator 3-fold), dilatih ±8 jt baris** | **3,088** | **1,516** | **0,892** |

Hyperparameter terbaik: `maxDepth=6, maxIter=100, stepSize=0.1`.

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
| `MODEL_TRAIN_ROWS` / `TUNING_ROWS` / `FINAL_TRAIN_ROWS` | 2 jt / 300 rb / 8 jt | ukuran data latih tiap tahap pemodelan |

> Catatan: `CrossValidator` sengaja memakai `parallelism=1`. Dengan `parallelism>1`, thread pool PySpark menahan
> referensi ke ribuan file *shuffle* GBT sehingga disk penuh (detail di laporan §4.6).
