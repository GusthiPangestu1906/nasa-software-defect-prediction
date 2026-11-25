import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.io import arff

# Menyesuaikan path agar bisa dijalankan dari root proyek atau dari dalam folder src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent if current_dir.name == "src" else current_dir
data_dir = project_root / "data"

arff_files = sorted(data_dir.glob("*.arff"))
print(f"Direktori data: {data_dir}")
print(f"File ARFF ditemukan: {[f.name for f in arff_files]}\n")

audit_results = []

for file in arff_files:
    try:
        data, meta = arff.loadarff(file)
        df = pd.DataFrame(data)
        
        # Decode byte strings ke string utf-8 jika ada (ciri khas scipy loadarff)
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            
        target_col = df.columns[-1]
        target_dist = df[target_col].value_counts(dropna=False).to_dict()
        
        loc_cols = [c for c in df.columns if "loc" in c.lower()]
        loc_info = "N/A"
        if loc_cols:
            loc_col = loc_cols[0]
            zero_loc = (pd.to_numeric(df[loc_col], errors="coerce") <= 0).sum()
            loc_info = f"{loc_col} (<=0: {zero_loc})"
            
        print(f"=== AUDIT: {file.name} ===")
        print(f"Shape        : {df.shape[0]} baris x {df.shape[1]} kolom")
        print(f"Missing vals : {df.isna().sum().sum()}")
        print(f"Duplikat     : {df.duplicated().sum()}")
        print(f"Target ({target_col}): {target_dist}")
        print(f"Kolom LOC    : {loc_info}")
        print("-" * 40)
        
        audit_results.append({
            "File": file.name,
            "Baris": df.shape[0],
            "Kolom": df.shape[1],
            "Missing": df.isna().sum().sum(),
            "Duplikat": df.duplicated().sum(),
            "Target_Col": target_col,
            "Target_Dist": str(target_dist)
        })
    except Exception as e:
        print(f"=== AUDIT: {file.name} GAGAL/KOSONG ===")
        print(f"Pesan Error  : {e}")
        print("-" * 40)

if audit_results:
    summary_df = pd.DataFrame(audit_results)
    print("\n=== RINGKASAN DATASET VALID ===")
    print(summary_df[["File", "Baris", "Kolom", "Missing", "Duplikat", "Target_Col"]].to_string(index=False))
