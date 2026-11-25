"""Modul preprocessing data bebas data leakage untuk NASA Software Defect Prediction."""
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = {"id", "module_id", "name", "filename", "unnamed: 0"}


def load_nasa_arff(filepath: Union[str, Path]) -> Tuple[pd.DataFrame, str]:
    """Memuat file ARFF NASA MDP dan mengembalikan DataFrame serta nama kolom target."""
    data, _ = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    target_col = df.columns[-1]
    return df, target_col


def clean_data(
    df: pd.DataFrame, target_col: Optional[str] = None, loc_col: str = "LOC_TOTAL"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Membersihkan duplikasi, kolom ID, nilai infinity, modul dengan LOC <= 0, dan konversi target Y/N -> 1/0."""
    cleaned_df = df.drop_duplicates().copy()

    if target_col is None:
        target_col = cleaned_df.columns[-1]

    # Hapus kolom ID jika ada agar tidak masuk sebagai fitur
    drop_cols = [c for c in cleaned_df.columns if c.lower() in ID_COLUMNS and c != target_col]
    if drop_cols:
        cleaned_df = cleaned_df.drop(columns=drop_cols)

    # Konversi target menjadi biner 1/0
    y_raw = cleaned_df[target_col]
    if not pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(str).str.strip().str.upper().map(
            {"Y": 1, "YES": 1, "TRUE": 1, "1": 1, "N": 0, "NO": 0, "FALSE": 0, "0": 0}
        )
    else:
        y = pd.to_numeric(y_raw, errors="coerce")

    X = cleaned_df.drop(columns=[target_col]).apply(pd.to_numeric, errors="coerce")

    # Ganti nilai infinity menjadi NaN agar bisa ditangani oleh imputer atau dibersihkan
    X = X.replace([np.inf, -np.inf], np.nan)

    # Hapus baris yang targetnya NaN, seluruh fiturnya NaN, atau LOC <= 0 (tidak valid untuk budget LOC)
    valid_mask = y.notna() & X.notna().any(axis=1)
    if loc_col in X.columns:
        valid_mask = valid_mask & (X[loc_col] > 0)

    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).reset_index(drop=True)

    return X, y


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float = 0.60,
    val_size: float = 0.20,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Memisahkan data menjadi Train, Validation, dan Test dengan stratified split dan seed tetap."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Jumlah proporsi train_size + val_size + test_size harus sama dengan 1.0")

    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, stratify=y, random_state=random_state
    )

    val_ratio_in_temp = test_size / temp_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_ratio_in_temp, stratify=y_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessing_pipeline(use_scaler: bool = True) -> Pipeline:
    """Membangun Pipeline imputer (dan scaler opsional) yang hanya di-fit pada data Train."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def build_model_pipeline(classifier, use_scaler: bool = True) -> Pipeline:
    """Membungkus imputer, scaler opsional, dan model dalam satu scikit-learn Pipeline."""
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", classifier))
    return Pipeline(steps)
