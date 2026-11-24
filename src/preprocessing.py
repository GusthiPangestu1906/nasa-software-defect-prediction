"""Modul preprocessing data bebas data leakage."""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def clean_data(df: pd.DataFrame, target_col: str = 'defects') -> pd.DataFrame:
    """Membersihkan duplikasi dan baris kosong dari dataset."""
    cleaned_df = df.drop_duplicates().copy()
    cleaned_df = cleaned_df.dropna()
    return cleaned_df

def split_features_target(
    df: pd.DataFrame, target_col: str = 'defects', test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Memisahkan fitur dan target dengan stratified split untuk menjaga rasio kelas."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Melakukan scaling fitur dengan fit hanya pada data training untuk mencegah leakage."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns), scaler
