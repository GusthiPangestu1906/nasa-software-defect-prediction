"""Skrip audit menyeluruh untuk dataset NASA Software Defect Prediction di folder data/."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import arff


def audit_directory(data_dir: Path) -> pd.DataFrame:
    """Mengaudit seluruh file CSV dan ARFF di direktori data/."""
    csv_files = sorted(data_dir.glob("*.csv"))
    arff_files = sorted(data_dir.glob("*.arff"))

    print(f"Direktori data : {data_dir}")
    print(f"File CSV       : {[f.name for f in csv_files]}")
    print(f"File ARFF      : {[f.name for f in arff_files]}\n")

    audit_results = []
    all_files = [(f, "csv") for f in csv_files] + [(f, "arff") for f in arff_files]

    for file, fmt in all_files:
        try:
            if fmt == "csv":
                df = pd.read_csv(file)
            else:
                data, _ = arff.loadarff(file)
                df = pd.DataFrame(data)
                for col in df.select_dtypes([object]).columns:
                    df[col] = df[col].apply(
                        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                    )

            if df.empty:
                print(f"=== AUDIT: {file.name} (KOSONG: {df.shape[0]}x{df.shape[1]}) ===")
                audit_results.append({
                    "File": file.name,
                    "Format": fmt.upper(),
                    "Baris": df.shape[0],
                    "Kolom": df.shape[1],
                    "Target_Col": df.columns[-1] if len(df.columns) > 0 else "N/A",
                    "LOC_Col": "N/A",
                    "Missing_NaN": 0,
                    "Infinity": 0,
                    "LOC_le_0": 0,
                    "Duplikat": 0,
                    "Defective_N": 0,
                    "NonDefective_N": 0,
                    "Defect_Pct": 0.0,
                    "Imbalance_Ratio": "N/A",
                })
                continue

            target_col = df.columns[-1]
            feature_df = df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")

            nan_count = int(df.isna().sum().sum())
            inf_count = int(np.isinf(feature_df.values).sum())
            dup_count = int(df.duplicated().sum())

            vc = df[target_col].astype(str).str.strip().str.upper().value_counts().to_dict()
            pos_count = vc.get("Y", 0) + vc.get("TRUE", 0) + vc.get("1", 0)
            neg_count = vc.get("N", 0) + vc.get("FALSE", 0) + vc.get("0", 0)
            total_valid = pos_count + neg_count
            defect_pct = (pos_count / total_valid * 100.0) if total_valid > 0 else 0.0
            imb_ratio = f"1:{neg_count / pos_count:.2f}" if pos_count > 0 else "N/A"

            loc_col = (
                "LOC_TOTAL"
                if "LOC_TOTAL" in df.columns
                else next((c for c in df.columns if "loc" in c.lower()), "N/A")
            )
            loc_le_0 = (
                int((pd.to_numeric(df[loc_col], errors="coerce") <= 0).sum())
                if loc_col != "N/A"
                else 0
            )

            print(f"=== AUDIT: {file.name} ===")
            print(f"Shape           : {df.shape[0]} baris x {df.shape[1]} kolom")
            print(f"Target Col      : {target_col} -> Y={pos_count}, N={neg_count} ({defect_pct:.2f}%, rasio {imb_ratio})")
            print(f"LOC Col         : {loc_col} (<= 0: {loc_le_0} baris)")
            print(f"Missing (NaN)   : {nan_count}")
            print(f"Infinity        : {inf_count}")
            print(f"Duplikat        : {dup_count}")
            print("-" * 45)

            audit_results.append({
                "File": file.name,
                "Format": fmt.upper(),
                "Baris": df.shape[0],
                "Kolom": df.shape[1],
                "Target_Col": target_col,
                "LOC_Col": loc_col,
                "Missing_NaN": nan_count,
                "Infinity": inf_count,
                "LOC_le_0": loc_le_0,
                "Duplikat": dup_count,
                "Defective_N": pos_count,
                "NonDefective_N": neg_count,
                "Defect_Pct": round(defect_pct, 2),
                "Imbalance_Ratio": imb_ratio,
            })
        except Exception as e:
            print(f"=== AUDIT: {file.name} ERROR: {e} ===")

    return pd.DataFrame(audit_results)


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent if current_dir.name == "src" else current_dir
    summary_df = audit_directory(project_root / "data")
    if not summary_df.empty:
        print("\n=== RINGKASAN AUDIT DATASET ===")
        print(summary_df.to_string(index=False))
