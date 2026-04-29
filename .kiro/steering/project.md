# CardioSense AI — MLOps Project

## What This Project Does
CardioSense AI is a heart disease prediction MLOps pipeline built on the UCI Cleveland Heart Disease dataset (303 patients, 13 clinical features). It trains and compares Random Forest and Gradient Boosting classifiers, tracks experiments with MLflow, serves predictions via a Flask REST API, and includes full unit tests.

## Architecture
```
cardiosense_mlops/
├── data/heart.csv              # UCI Cleveland dataset (303 rows, 14 cols)
├── src/
│   ├── preprocess.py           # Data loading, validation, scaling
│   ├── train.py                # Model training + MLflow logging
│   ├── evaluate.py             # Metrics, drift detection, report
│   └── pipeline.py             # Master orchestrator
├── api/app.py                  # Flask REST API (predict, health, metrics)
├── models/                     # Saved .pkl models + evaluation_report.json
├── tests/test_pipeline.py      # Pytest unit tests
├── mlflow.db                   # SQLite MLflow tracking store
├── mlruns/                     # MLflow artifact store
├── run_pipeline.py             # Root-level entry point (run this!)
├── requirements.txt
└── Dockerfile
```

## Key Entry Points
- **Run full pipeline:** `python run_pipeline.py`
- **Run tests:** `pytest tests/ -v`
- **Launch MLflow UI:** `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`
- **Start API server:** `python api/app.py`
- **Docker build:** `docker build -t cardiosense:latest .`

## MLflow Setup
- Backend store: `sqlite:///mlflow.db` (file at project root)
- Artifact store: `./mlruns`
- Experiment name: `CardioSense_AI`
- Tracking URI set via `MLFLOW_TRACKING_URI` env var or defaulted in code

## Models
- **RandomForest** — n_estimators=200, criterion=gini (typically best by AUC-ROC)
- **GradientBoosting** — n_estimators=100, lr=0.1, max_depth=3

## Clinical Thresholds (production-readiness)
| Metric | Minimum |
|--------|---------|
| Sensitivity (Recall) | ≥ 0.80 |
| Specificity | ≥ 0.75 |
| AUC-ROC | ≥ 0.85 |
