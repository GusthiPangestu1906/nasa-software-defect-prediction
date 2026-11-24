# Cost-Aware Software Defect Prediction Using Static Code Metrics and Machine Learning

Repositori ini merupakan kerangka kerja komputasi dan eksperimen untuk Tugas Akhir mata kuliah Tata Tulis Karya Ilmiah (TTKI) - Topik P3.

## Research Questions & Tujuan
1. **RQ1**: Bagaimana pengaruh penerapan teknik penyeimbangan kelas (*class weighting*) terhadap nilai PR-AUC dan Recall pada model Logistic Regression dan Random Forest?
2. **RQ2**: Bagaimana efektivitas pemeringkatan risiko berbasis probabilitas prediksi model dalam mendeteksi defect pada batas usaha inspeksi 20% total LOC (*Recall@20% LOC*)?

## Alur Kerja Penelitian
`mermaid
flowchart TD
    A[Topik & Literatur] --> B[Research Gap & RQ]
    B --> C[Audit Dataset Kaggle NASA SDP]
    C --> D[Reproduksi Baseline & Eksperimen]
    D --> E[Evaluasi & Penulisan IMRAD]
    E --> F[Peer Review & Rilis Repositori]
`

## Sumber Data & Repositori Baseline
- **Dataset**: NASA Software Defect Prediction (Kaggle: https://www.kaggle.com/datasets/aczy156/nasa-software-defect-prediction)
- **Baseline Reference**: Repositori GitHub SonamKumari1227 (https://github.com/SonamKumari1227/-Software-Defect-Prediction)

## Setup Environment
Jalankan instruksi berikut di terminal:

`ash
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
# Di Windows:
.venv\Scripts\activate
# Di Linux/macOS:
source .venv/bin/activate

# 2. Instalasi dependensi
pip install -r requirements.txt

# 3. Kunci versi environment
pip freeze > environment-freeze.txt
`

## Urutan Eksekusi Notebook
Setiap eksperimen menggunakan random seed tetap (seed=42) dan seluruh preprocessing di-fit hanya pada data training:
1. 
otebooks/01_eda.ipynb: Audit data, analisis korelasi fitur, missing values, dan distribusi kelas.
2. 
otebooks/02_baseline.ipynb: Pelatihan model Logistic Regression dan Random Forest standar (tanpa bobot kelas).
3. 
otebooks/03_experiment.ipynb: Pengujian skenario class weighting, pemeringkatan probabilitas risiko, dan metrik Cost-Aware (Recall@20% LOC).

## Struktur Repositori
`	ext
ttki-p3-software-defect-prediction/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_experiment.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── evaluation.py
├── results/
├── figures/
├── paper/
└── references/
    └── literature_matrix.csv
`
