"""Modul kalkulasi metrik evaluasi standar dan cost-aware defect prediction."""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Menghitung metrik performa klasifikasi termasuk PR-AUC (Average Precision)."""
    pr_auc = average_precision_score(y_true, y_prob)
    return {
        "pr_auc": float(pr_auc),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def calculate_recall_at_20_percent_loc(
    y_true: np.ndarray,
    scores: np.ndarray,
    loc: np.ndarray,
    budget_ratio: float = 0.20,
) -> float:
    """Menghitung proporsi defect ditemukan saat inspeksi mencapai batas budget LOC (default 20%)."""
    eval_df = pd.DataFrame(
        {
            "defect": np.asarray(y_true),
            "score": np.asarray(scores),
            "loc": np.asarray(loc, dtype=float),
        }
    )
    eval_df = eval_df.sort_values(by="score", ascending=False, kind="mergesort").reset_index(drop=True)

    total_loc = eval_df["loc"].sum()
    total_defects = eval_df["defect"].sum()
    if total_defects == 0 or total_loc == 0:
        return 0.0

    eval_df["cum_loc"] = eval_df["loc"].cumsum()
    cutoff_loc = budget_ratio * total_loc

    inspected = eval_df[eval_df["cum_loc"] <= cutoff_loc]
    found_defects = inspected["defect"].sum() if not inspected.empty else 0
    return float(found_defects / total_defects)


def compute_loc_recall_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    loc: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Menghasilkan titik kurva kumulatif (% LOC diinspeksi, % Defect ditemukan) untuk visualisasi."""
    eval_df = pd.DataFrame(
        {
            "defect": np.asarray(y_true),
            "score": np.asarray(scores),
            "loc": np.asarray(loc, dtype=float),
        }
    )
    eval_df = eval_df.sort_values(by="score", ascending=False, kind="mergesort").reset_index(drop=True)

    total_loc = eval_df["loc"].sum()
    total_defects = eval_df["defect"].sum()
    if total_loc == 0 or total_defects == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    cum_loc_pct = np.concatenate([[0.0], eval_df["loc"].cumsum().values / total_loc])
    cum_defect_pct = np.concatenate([[0.0], eval_df["defect"].cumsum().values / total_defects])
    return cum_loc_pct, cum_defect_pct


def get_pr_curve_points(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Mengembalikan array (recall, precision) untuk menggambar Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return recall, precision
