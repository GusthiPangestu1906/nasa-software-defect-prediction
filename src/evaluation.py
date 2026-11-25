"""Modul kalkulasi metrik evaluasi standar dan cost-aware defect prediction."""
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Menghitung metrik performa klasifikasi termasuk PR-AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return {
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'pr_auc': float(pr_auc)
    }

def calculate_recall_at_20_percent_loc(y_true: np.ndarray, y_prob: np.ndarray, loc: np.ndarray) -> float:
    """Menghitung proporsi defect yang ditemukan saat inspeksi mencapai 20% dari total LOC."""
    df = pd.DataFrame({'defect': y_true, 'prob': y_prob, 'loc': loc})
    df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
    
    total_loc = df['loc'].sum()
    total_defects = df['defect'].sum()
    if total_defects == 0 or total_loc == 0:
        return 0.0

    df['cum_loc'] = df['loc'].cumsum()
    df['cum_defects'] = df['defect'].cumsum()
    cutoff_loc = 0.20 * total_loc
    
    inspected = df[df['cum_loc'] <= cutoff_loc]
    found_defects = inspected['cum_defects'].iloc[-1] if not inspected.empty else 0
    return float(found_defects / total_defects)
