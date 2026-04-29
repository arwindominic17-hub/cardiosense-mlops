"""
pipeline.py
===========
Master orchestrator for the CardioSense AI MLOps pipeline.
Runs: ingest → validate → preprocess → train → evaluate → register
"""

import json
import logging
import os
import sys

import joblib
import mlflow
import numpy as np

# Add src to path so local modules resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from evaluate import detect_drift, evaluate_model, generate_evaluation_report  # noqa: E402
from preprocess import load_data, preprocess, validate_data  # noqa: E402
from train import train_and_log  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ──
# Resolve project root (one level above src/)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)

DATA_PATH = os.environ.get("DATA_PATH", os.path.join(_ROOT_DIR, "data", "heart.csv"))
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(_ROOT_DIR, "models"))
MLFLOW_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(_ROOT_DIR, 'mlflow.db')}"
)
EXPERIMENT_NAME = os.environ.get("EXPERIMENT", "CardioSense_AI")
TEST_SIZE = float(os.environ.get("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))


def run_pipeline():
    log.info("=" * 60)
    log.info("  CardioSense AI — MLOps Pipeline Starting")
    log.info("=" * 60)

    mlflow.set_tracking_uri(MLFLOW_URI)
    log.info(f"  MLflow tracking URI: {MLFLOW_URI}")

    # ── STEP 1: Ingest & Validate ──
    log.info("\n[STEP 1] Data Ingestion & Validation")
    df = load_data(DATA_PATH)

    with mlflow.start_run(
        run_name="data_validation",
        nested=False,
        experiment_id=mlflow.set_experiment(EXPERIMENT_NAME).experiment_id,
    ):
        report = validate_data(df)
        log.info(
            f"  Rows: {len(df)} | Missing: {report['total_missing_values']} | "
            f"Duplicates: {report['duplicate_rows']} | "
            f"Class balance: {report['class_balance_ratio']}"
        )

    # ── STEP 2: Preprocess ──
    log.info("\n[STEP 2] Preprocessing")
    X_train, X_test, y_train, y_test, scaler = preprocess(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ── STEP 3: Train + Track ──
    log.info("\n[STEP 3] Model Training with MLflow Tracking")
    best_model, best_metrics, best_run_id, best_name = train_and_log(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler=scaler,
        experiment_name=EXPERIMENT_NAME,
        models_dir=MODELS_DIR,
        mlflow_uri=MLFLOW_URI,
    )

    # ── STEP 4: Evaluate ──
    log.info("\n[STEP 4] Model Evaluation")
    # Load the saved bundle to get the optimized threshold
    bundle_path = os.path.join(MODELS_DIR, "production_bundle.pkl")
    bundle = joblib.load(bundle_path)
    best_threshold = float(bundle.get("threshold", 0.5))
    log.info(f"  Using decision threshold: {best_threshold}")

    eval_metrics = evaluate_model(best_model, X_test, y_test, best_name, threshold=best_threshold)

    # Load feature importance from saved file
    fi_path = f"{MODELS_DIR}/{best_name}_feature_importance.json"
    fi = {}
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            fi = json.load(f)

    # Drift detection: simulate new data = test set with small noise
    X_new = X_test.copy()
    X_new["age"] = X_new["age"] + np.random.normal(0, 0.1, len(X_new))
    drift = detect_drift(X_train, X_new, threshold=0.15)

    report_path = generate_evaluation_report(
        eval_metrics,
        best_name,
        fi,
        drift,
        output_path=os.path.join(MODELS_DIR, "evaluation_report.json"),
    )

    # ── STEP 5: Summary ──
    log.info("\n" + "=" * 60)
    log.info("  PIPELINE COMPLETE — SUMMARY")
    log.info("=" * 60)
    log.info(f"  Best Model:   {best_name}")
    log.info(
        f"  Accuracy:     {eval_metrics['accuracy']:.4f}  ({eval_metrics['accuracy'] * 100:.1f}%)"
    )
    log.info(f"  AUC-ROC:      {eval_metrics['auc_roc']:.4f}")
    log.info(f"  F1-Score:     {eval_metrics['f1_score']:.4f}")
    log.info(f"  Sensitivity:  {eval_metrics['sensitivity']:.4f}")
    log.info(f"  Specificity:  {eval_metrics['specificity']:.4f}")
    log.info(f"  MLflow Run:   {best_run_id}")
    log.info(f"  Report:       {report_path}")
    log.info("=" * 60)

    return best_model, eval_metrics


if __name__ == "__main__":
    run_pipeline()
