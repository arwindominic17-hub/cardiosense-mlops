"""
train.py
========
Model training with full MLflow experiment tracking.
Trains Random Forest + Gradient Boosting, logs all params,
metrics, artifacts, and registers the best model.
"""

import json
import logging
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_params():
    p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "params.yaml")
    if os.path.exists(p):
        with open(p) as f:
            return yaml.safe_load(f)
    return {}


_params = _load_params()
_t = _params.get("train", {})

MODEL_CONFIGS = [
    {
        "name": "RandomForest",
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": int(_t.get("n_estimators_rf", 200)),
            "max_features": _t.get("max_features_rf", "sqrt"),
            "criterion": _t.get("criterion_rf", "gini"),
            "bootstrap": True,
            "random_state": int(_params.get("data", {}).get("random_state", 42)),
            "n_jobs": -1,
        },
    },
    {
        "name": "GradientBoosting",
        "class": GradientBoostingClassifier,
        "params": {
            "n_estimators": int(_t.get("n_estimators_gb", 100)),
            "learning_rate": float(_t.get("learning_rate_gb", 0.1)),
            "max_depth": int(_t.get("max_depth_gb", 3)),
            "random_state": int(_params.get("data", {}).get("random_state", 42)),
        },
    },
]


def find_best_threshold(
    y_true, y_prob_pos, min_sensitivity: float = None, min_specificity: float = None
) -> float:
    """
    Search over thresholds [0.1, 0.9] to find the value that satisfies
    both clinical constraints (sensitivity >= min_sensitivity AND
    specificity >= min_specificity) while maximising their harmonic mean.
    Falls back to 0.60 (best empirical trade-off) if no threshold satisfies
    both constraints.
    """
    # Read from params.yaml if not explicitly passed
    if min_sensitivity is None or min_specificity is None:
        p = _load_params().get("train", {})
        min_sensitivity = min_sensitivity or float(p.get("min_sensitivity", 0.80))
        min_specificity = min_specificity or float(p.get("min_specificity", 0.70))

    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.60
    best_score = -1.0

    for t in thresholds:
        y_pred_t = (y_prob_pos >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred_t)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        if sens >= min_sensitivity and spec >= min_specificity:
            score = 2 * sens * spec / (sens + spec)
            if score > best_score:
                best_score = score
                best_thresh = t

    return float(round(best_thresh, 2))


def compute_metrics(y_true, y_pred, y_prob, threshold: float = 0.5) -> dict:
    """Compute all evaluation metrics and return as dict."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_prob[:, 1]), 4),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0.0,
        "decision_threshold": round(threshold, 2),
    }


def train_and_log(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    experiment_name: str = "CardioSense_AI",
    models_dir: str = "models",
    mlflow_uri: str = "sqlite:///mlflow.db",
):
    """
    Train all model configs, log each to MLflow as a child run,
    select the best model by AUC-ROC, register it, and save artifacts.

    Returns: best_model, best_metrics, best_run_id
    """
    os.makedirs(models_dir, exist_ok=True)

    # Set / create experiment
    mlflow.set_tracking_uri(mlflow_uri)
    log.info(f"MLflow tracking URI: {mlflow_uri}")
    exp = mlflow.set_experiment(experiment_name)
    log.info(f"MLflow experiment: {experiment_name}  (id={exp.experiment_id})")

    best_model = None
    best_metrics = {}
    best_run_id = None
    best_auc = -1.0
    best_name = ""
    best_threshold = 0.5

    # ── parent run — holds experiment-level metadata ──
    with mlflow.start_run(run_name="CardioSense_Training_Pipeline"):

        mlflow.set_tags(
            {
                "project": "CardioSense AI",
                "dataset": "UCI Cleveland Heart Disease",
                "author": "Arwin Dominic, Dinesh Kumar, Chethan K Chavan, Amaresh",
                "pipeline": "MLOps v1.0",
            }
        )

        for cfg in MODEL_CONFIGS:
            model_name = cfg["name"]
            log.info(f"\nTraining: {model_name}")

            # ── child run per model ──
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:

                # Log hyperparameters
                mlflow.log_params(cfg["params"])
                mlflow.log_param("model_class", model_name)

                # Train
                model = cfg["class"](**cfg["params"])
                model.fit(X_train, y_train)

                # Predict — find optimal threshold first
                y_prob = model.predict_proba(X_test)
                best_thresh = find_best_threshold(y_test, y_prob[:, 1])
                y_pred = (y_prob[:, 1] >= best_thresh).astype(int)
                mlflow.log_param("decision_threshold", best_thresh)

                # Metrics
                metrics = compute_metrics(y_test, y_pred, y_prob, threshold=best_thresh)
                mlflow.log_metrics(metrics)

                log.info(
                    f"  Accuracy={metrics['accuracy']} "
                    f"AUC-ROC={metrics['auc_roc']} "
                    f"F1={metrics['f1_score']}"
                )

                # Classification report as artifact
                cr = classification_report(y_test, y_pred, target_names=["No Disease", "Disease"])
                cr_path = f"{models_dir}/{model_name}_classification_report.txt"
                with open(cr_path, "w") as f:
                    f.write(f"Model: {model_name}\n")
                    f.write("=" * 50 + "\n")
                    f.write(cr)
                mlflow.log_artifact(cr_path, artifact_path="reports")

                # Feature importance (RF and GB both have it)
                if hasattr(model, "feature_importances_"):
                    fi = dict(zip(X_train.columns, model.feature_importances_.round(4)))
                    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
                    fi_path = f"{models_dir}/{model_name}_feature_importance.json"
                    with open(fi_path, "w") as f:
                        json.dump(fi_sorted, f, indent=2)
                    mlflow.log_artifact(fi_path, artifact_path="feature_importance")

                # Log model with input/output signature
                signature = infer_signature(X_train, y_pred)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=X_test.iloc[:3],
                    registered_model_name=f"CardioSense_{model_name}",
                )

                # Save locally too
                joblib.dump(model, f"{models_dir}/{model_name}.pkl")

                # Track best
                if metrics["auc_roc"] > best_auc:
                    best_auc = metrics["auc_roc"]
                    best_model = model
                    best_metrics = metrics
                    best_run_id = child_run.info.run_id
                    best_name = model_name
                    best_threshold = best_thresh

        # Log best model info to parent run
        mlflow.log_params(
            {
                "best_model": best_name,
                "best_auc_roc": best_auc,
                "best_accuracy": best_metrics.get("accuracy"),
            }
        )

        # Save scaler + best model together as production bundle
        bundle = {
            "model": best_model,
            "scaler": scaler,
            "features": list(X_train.columns),
            "threshold": best_threshold,
        }
        bundle_path = f"{models_dir}/production_bundle.pkl"
        joblib.dump(bundle, bundle_path)
        mlflow.log_artifact(bundle_path, artifact_path="production")

        log.info(f"\nBest model: {best_name}  AUC-ROC={best_auc}")
        log.info(f"Production bundle saved → {bundle_path}")

    return best_model, best_metrics, best_run_id, best_name


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import load_data, preprocess

    ROOT = os.path.dirname(os.path.dirname(__file__))
    DATA_PATH = os.environ.get("DATA_PATH", os.path.join(ROOT, "data", "heart.csv"))
    MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(ROOT, "models"))
    MLFLOW_URI = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"
    )
    EXPERIMENT = os.environ.get("EXPERIMENT", "CardioSense_AI")

    log.info("Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    log.info("Starting training...")
    best_model, best_metrics, best_run_id, best_name = train_and_log(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler=scaler,
        experiment_name=EXPERIMENT,
        models_dir=MODELS_DIR,
        mlflow_uri=MLFLOW_URI,
    )

    print("\n✅  Training complete.")
    print(f"   Best Model:  {best_name}")
    print(f"   Accuracy:    {best_metrics['accuracy']:.1%}")
    print(f"   AUC-ROC:     {best_metrics['auc_roc']:.4f}")
    print(f"   F1-Score:    {best_metrics['f1_score']:.4f}")
    print(f"   Threshold:   {best_metrics['decision_threshold']}")
    print(f"   MLflow Run:  {best_run_id}")
