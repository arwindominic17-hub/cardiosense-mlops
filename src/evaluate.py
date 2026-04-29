"""
evaluate.py
===========
Post-training evaluation, model comparison, and data drift detection.
Generates a detailed evaluation report and logs everything to MLflow.
"""

import json
import logging
import os

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def evaluate_model(
    model, X_test, y_test, model_name: str, run_id: str = None, threshold: float = 0.5
) -> dict:
    """Run full evaluation on test set and return metrics dict."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_prob), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "total_test": int(len(y_test)),
        "correct": int(tp + tn),
        "decision_threshold": round(threshold, 2),
    }

    log.info(
        f"[{model_name}] Accuracy={metrics['accuracy']} "
        f"AUC={metrics['auc_roc']} F1={metrics['f1_score']} "
        f"Sensitivity={metrics['sensitivity']} Specificity={metrics['specificity']} "
        f"Threshold={threshold}"
    )
    return metrics


def detect_drift(X_train: pd.DataFrame, X_new: pd.DataFrame, threshold: float = 0.15) -> dict:
    """
    Simple statistical drift detection using mean shift.
    Flags any feature where the mean has shifted by more than
    `threshold` standard deviations relative to the training distribution.
    Returns a dict of {feature: drift_detected (bool)}.
    """
    drift_report = {}
    train_means = X_train.mean()
    train_stds = X_train.std().replace(0, 1e-9)
    new_means = X_new.mean()

    for col in X_train.columns:
        shift = abs((new_means[col] - train_means[col]) / train_stds[col])
        drift_report[col] = {
            "train_mean": round(float(train_means[col]), 4),
            "new_mean": round(float(new_means[col]), 4),
            "std_shift": round(float(shift), 4),
            "drift": bool(shift > threshold),  # Python bool, not numpy bool
        }

    drifted = [k for k, v in drift_report.items() if v["drift"]]
    log.info(f"Drift detected in {len(drifted)}/{len(X_train.columns)} features: {drifted}")

    if mlflow.active_run():
        mlflow.log_metric("drifted_features_count", len(drifted))
        mlflow.log_param("drift_threshold", threshold)

    return drift_report


def generate_evaluation_report(
    metrics: dict,
    model_name: str,
    feature_importance: dict,
    drift_report: dict,
    output_path: str = "models/evaluation_report.json",
) -> str:
    """Write a comprehensive JSON evaluation report and log to MLflow."""
    report = {
        "model": model_name,
        "performance": metrics,
        "feature_importance_top5": dict(
            list(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))[:5]
        ),
        "drift_summary": {
            "drifted_features": [k for k, v in drift_report.items() if v.get("drift") is True],
            "clean_features": [k for k, v in drift_report.items() if not v.get("drift")],
        },
        "clinical_thresholds": {
            "min_sensitivity": 0.80,
            "min_specificity": 0.70,
            "min_auc": 0.85,
            "sensitivity_ok": bool(metrics.get("sensitivity", 0) >= 0.80),
            "specificity_ok": bool(metrics.get("specificity", 0) >= 0.70),
            "auc_ok": bool(metrics.get("auc_roc", 0) >= 0.85),
            "ready_for_prod": bool(
                all(
                    [
                        metrics.get("sensitivity", 0) >= 0.80,
                        metrics.get("specificity", 0) >= 0.70,
                        metrics.get("auc_roc", 0) >= 0.85,
                    ]
                )
            ),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    if mlflow.active_run():
        mlflow.log_artifact(output_path, artifact_path="evaluation")
        mlflow.log_metric("prod_ready", int(report["clinical_thresholds"]["ready_for_prod"]))

    log.info(f"Evaluation report → {output_path}")
    log.info(f"Production ready: {report['clinical_thresholds']['ready_for_prod']}")
    return output_path
