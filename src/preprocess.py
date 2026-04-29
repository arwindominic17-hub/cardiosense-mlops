"""
preprocess.py
=============
Data loading, validation, and preprocessing pipeline for CardioSense AI.
Part of the MLOps integration — all steps are logged to MLflow.
"""

import logging

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COL = "target"

EXPECTED_RANGES = {
    "age": (20, 100),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "thalach": (60, 210),
    "oldpeak": (0.0, 7.0),
    "ca": (0, 4),
}


def load_data(path: str) -> pd.DataFrame:
    """Load CSV and do basic sanity checks."""
    log.info(f"Loading data from: {path}")
    df = pd.read_csv(path)

    # Validate columns
    missing = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    log.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def validate_data(df: pd.DataFrame) -> dict:
    """
    Run data quality checks and return a report dict.
    All metrics are logged to the active MLflow run.
    """
    report = {}

    # Check for missing columns first
    missing_cols = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Missing values
    missing_pct = df[FEATURE_COLS + [TARGET_COL]].isnull().mean().to_dict()
    report["missing_pct"] = missing_pct
    total_missing = df[FEATURE_COLS + [TARGET_COL]].isnull().sum().sum()
    report["total_missing_values"] = int(total_missing)

    # Class distribution
    vc = df[TARGET_COL].value_counts().to_dict()
    report["class_0_count"] = int(vc.get(0, 0))
    report["class_1_count"] = int(vc.get(1, 0))
    report["class_balance_ratio"] = round(vc.get(1, 0) / max(vc.get(0, 1), 1), 4)

    # Range checks
    out_of_range = {}
    for col, (lo, hi) in EXPECTED_RANGES.items():
        if col in df.columns:
            n_oob = int(((df[col] < lo) | (df[col] > hi)).sum())
            out_of_range[col] = n_oob
    report["out_of_range_counts"] = out_of_range
    report["total_out_of_range"] = sum(out_of_range.values())

    # Duplicate rows
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Log to MLflow if a run is active
    if mlflow.active_run():
        mlflow.log_metrics(
            {
                "data_total_rows": len(df),
                "data_missing_total": report["total_missing_values"],
                "data_class_0": report["class_0_count"],
                "data_class_1": report["class_1_count"],
                "data_balance_ratio": report["class_balance_ratio"],
                "data_out_of_range": report["total_out_of_range"],
                "data_duplicate_rows": report["duplicate_rows"],
            }
        )

    log.info(
        f"Data validation: {report['total_missing_values']} missing, "
        f"{report['duplicate_rows']} duplicates, "
        f"class balance {report['class_balance_ratio']:.2f}"
    )
    return report


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline:
      1. Drop duplicates
      2. Split features / target
      3. Stratified train/test split
      4. StandardScaler fit on train, transform both
    Returns X_train, X_test, y_train, y_test, scaler
    """
    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info(f"Dropped {before - len(df)} duplicate rows")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info(f"Split: {len(X_train)} train / {len(X_test)} test")

    # StandardScaler — fit ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Log split info to MLflow
    if mlflow.active_run():
        mlflow.log_params(
            {
                "test_size": test_size,
                "random_state": random_state,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "n_features": len(FEATURE_COLS),
                "scaler": "StandardScaler",
            }
        )

    return (
        pd.DataFrame(X_train_scaled, columns=FEATURE_COLS),
        pd.DataFrame(X_test_scaled, columns=FEATURE_COLS),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        scaler,
    )
