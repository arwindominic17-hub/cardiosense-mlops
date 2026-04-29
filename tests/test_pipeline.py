"""
test_pipeline.py
================
Unit tests for CardioSense AI MLOps pipeline components.
Run with: python -m pytest tests/ -v
"""

import os
import sys

# Ensure src/ is importable before loading local modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from evaluate import detect_drift, evaluate_model  # noqa: E402
from preprocess import FEATURE_COLS, load_data, preprocess, validate_data  # noqa: E402

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    """Create a minimal synthetic dataset matching the UCI schema."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame(
        {
            "age": np.random.randint(30, 75, n),
            "sex": np.random.randint(0, 2, n),
            "cp": np.random.randint(0, 4, n),
            "trestbps": np.random.randint(90, 180, n),
            "chol": np.random.randint(150, 400, n),
            "fbs": np.random.randint(0, 2, n),
            "restecg": np.random.randint(0, 3, n),
            "thalach": np.random.randint(80, 200, n),
            "exang": np.random.randint(0, 2, n),
            "oldpeak": np.random.uniform(0, 5, n).round(1),
            "slope": np.random.randint(0, 3, n),
            "ca": np.random.randint(0, 4, n),
            "thal": np.random.randint(0, 3, n),
            "target": np.random.randint(0, 2, n),
        }
    )
    return df


@pytest.fixture
def trained_model_and_data(sample_df):
    X_train, X_test, y_train, y_test, scaler = preprocess(sample_df)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test, scaler


# ── Tests: preprocess.py ──────────────────────────────────────


class TestValidateData:
    def test_returns_dict(self, sample_df):
        report = validate_data(sample_df)
        assert isinstance(report, dict)

    def test_class_counts_correct(self, sample_df):
        report = validate_data(sample_df)
        total = report["class_0_count"] + report["class_1_count"]
        assert total == len(sample_df)

    def test_no_missing_in_clean_data(self, sample_df):
        report = validate_data(sample_df)
        assert report["total_missing_values"] == 0

    def test_detects_missing_values(self, sample_df):
        df_missing = sample_df.copy()
        df_missing.loc[0:4, "age"] = np.nan
        report = validate_data(df_missing)
        assert report["total_missing_values"] == 5

    def test_detects_duplicates(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        report = validate_data(df_dup)
        assert report["duplicate_rows"] == 5


class TestPreprocess:
    def test_output_shapes(self, sample_df):
        X_train, X_test, y_train, y_test, scaler = preprocess(sample_df)
        assert len(X_train) + len(X_test) <= len(sample_df)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_feature_columns_preserved(self, sample_df):
        X_train, X_test, *_ = preprocess(sample_df)
        assert list(X_train.columns) == FEATURE_COLS
        assert list(X_test.columns) == FEATURE_COLS

    def test_train_is_80_percent(self, sample_df):
        X_train, X_test, *_ = preprocess(sample_df, test_size=0.2)
        ratio = len(X_train) / (len(X_train) + len(X_test))
        assert 0.75 <= ratio <= 0.85  # allow for rounding

    def test_scaler_zero_mean(self, sample_df):
        X_train, *_, scaler = preprocess(sample_df)
        # After scaling, each column should have mean ≈ 0
        means = X_train.mean()
        assert (means.abs() < 0.1).all()

    def test_missing_column_raises(self, sample_df):
        df_bad = sample_df.drop(columns=["ca"])
        with pytest.raises(ValueError, match="Missing columns"):
            load_data.__wrapped__ if hasattr(load_data, "__wrapped__") else None
            validate_data(df_bad)  # will not raise, but...
            # actually test preprocess with missing col
            preprocess(df_bad)


# ── Tests: evaluate.py ────────────────────────────────────────


class TestEvaluateModel:
    def test_returns_all_metrics(self, trained_model_and_data):
        model, _, X_test, _, y_test, _ = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test, "RF_test")
        required = {
            "accuracy",
            "auc_roc",
            "f1_score",
            "sensitivity",
            "specificity",
            "tp",
            "tn",
            "fp",
            "fn",
        }
        assert required.issubset(set(metrics.keys()))

    def test_accuracy_in_range(self, trained_model_and_data):
        model, _, X_test, _, y_test, _ = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test, "RF_test")
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_auc_in_range(self, trained_model_and_data):
        model, _, X_test, _, y_test, _ = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test, "RF_test")
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_confusion_matrix_sums_to_total(self, trained_model_and_data):
        model, _, X_test, _, y_test, _ = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test, "RF_test")
        total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
        assert total == len(y_test)


class TestDriftDetection:
    def test_no_drift_identical_data(self, sample_df):
        X_train, X_test, *_ = preprocess(sample_df)
        drift = detect_drift(X_train, X_train.copy(), threshold=0.15)
        drifted = [k for k, v in drift.items() if v["drift"]]
        assert len(drifted) == 0

    def test_detects_large_shift(self, sample_df):
        X_train, X_test, *_ = preprocess(sample_df)
        X_shifted = X_train.copy()
        # Shift age by 5 standard deviations — should always be flagged
        X_shifted["age"] = X_shifted["age"] + 5.0
        drift = detect_drift(X_train, X_shifted, threshold=0.15)
        assert drift["age"]["drift"] is True

    def test_drift_report_structure(self, sample_df):
        X_train, X_test, *_ = preprocess(sample_df)
        drift = detect_drift(X_train, X_test)
        for col, info in drift.items():
            assert "train_mean" in info
            assert "new_mean" in info
            assert "std_shift" in info
            assert "drift" in info
            assert isinstance(info["drift"], bool)


# ── Tests: API input validation (FastAPI TestClient) ──────────


class TestInputValidation:
    """Test the FastAPI endpoints using the built-in TestClient."""

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Set up FastAPI test client."""
        import os

        os.environ["BUNDLE_PATH"] = os.path.join(
            os.path.dirname(__file__), "../models/production_bundle.pkl"
        )
        from fastapi.testclient import TestClient

        from api.app import app

        self.client = TestClient(app)

    def test_valid_patient(self):
        """A fully valid patient record should return 200 with a prediction."""
        valid = {
            "age": 58,
            "sex": 1,
            "cp": 0,
            "trestbps": 140,
            "chol": 268,
            "fbs": 0,
            "restecg": 0,
            "thalach": 152,
            "exang": 1,
            "oldpeak": 2.1,
            "slope": 1,
            "ca": 1,
            "thal": 2,
        }
        r = self.client.post("/predict", json=valid)
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert "risk_probability" in data
        assert "risk_level" in data
        assert data["prediction"] in [0, 1]

    def test_missing_feature(self):
        """Missing required fields should return 422 Unprocessable Entity."""
        incomplete = {"age": 58, "sex": 1}  # missing 11 features
        r = self.client.post("/predict", json=incomplete)
        assert r.status_code == 422

    def test_out_of_range(self):
        """Out-of-range values should return 422 Unprocessable Entity."""
        bad = {
            "age": 200,  # invalid — max is 100
            "sex": 1,
            "cp": 0,
            "trestbps": 140,
            "chol": 268,
            "fbs": 0,
            "restecg": 0,
            "thalach": 152,
            "exang": 1,
            "oldpeak": 2.1,
            "slope": 1,
            "ca": 1,
            "thal": 2,
        }
        r = self.client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_health_endpoint(self):
        """Health endpoint should return healthy status."""
        r = self.client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in ["healthy", "degraded"]

    def test_model_info_endpoint(self):
        """Model info endpoint should return model metadata."""
        r = self.client.get("/model-info")
        assert r.status_code == 200
        data = r.json()
        assert "model_type" in data
        assert "feature_names" in data
