"""
run_pipeline.py
===============
Root-level entry point for CardioSense AI MLOps pipeline.
Run from the project root:

    python run_pipeline.py

Environment variables (all optional — sensible defaults used):
    DATA_PATH             path to heart.csv       (default: data/heart.csv)
    MODELS_DIR            where to save .pkl files (default: models/)
    MLFLOW_TRACKING_URI   MLflow backend URI       (default: sqlite:///mlflow.db)
    EXPERIMENT            MLflow experiment name   (default: CardioSense_AI)
    TEST_SIZE             train/test split ratio   (default: 0.2)
    RANDOM_STATE          random seed              (default: 42)

After running, open the MLflow UI with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
"""

import os
import sys

# Ensure src/ is on the path regardless of where the user launches from
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

# Set defaults relative to project root (can be overridden via env vars)
os.environ.setdefault("DATA_PATH",            os.path.join(ROOT_DIR, "data", "heart.csv"))
os.environ.setdefault("MODELS_DIR",           os.path.join(ROOT_DIR, "models"))
os.environ.setdefault("MLFLOW_TRACKING_URI",  f"sqlite:///{os.path.join(ROOT_DIR, 'mlflow.db')}")
os.environ.setdefault("EXPERIMENT",           "CardioSense_AI")

from pipeline import run_pipeline  # noqa: E402

if __name__ == "__main__":
    model, metrics = run_pipeline()
    print("\n✅  Pipeline complete.")
    print(f"   Accuracy:    {metrics['accuracy']:.1%}")
    print(f"   AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"   F1-Score:    {metrics['f1_score']:.4f}")
    print(f"   Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"   Specificity: {metrics['specificity']:.4f}")
    print("\n📊  View results in MLflow UI:")
    print("   mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")
    print("   → http://localhost:5001")
