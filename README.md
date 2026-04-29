# CardioSense AI — MLOps Pipeline

Heart disease prediction pipeline with full MLflow experiment tracking, Flask REST API, and Docker support. Built on the UCI Cleveland Heart Disease dataset.

---

## Quick Start (Kiro / local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full ML pipeline
python run_pipeline.py

# 3. Open MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
# → http://localhost:5001

# 4. Start the prediction API
python api/app.py
# → http://localhost:5000/health
```

---

## Project Structure

```
cardiosense_mlops/
├── run_pipeline.py             ← Start here. Runs the full pipeline.
├── requirements.txt
├── Dockerfile
│
├── data/
│   └── heart.csv               UCI Cleveland dataset (303 patients, 14 cols)
│
├── src/
│   ├── pipeline.py             Master orchestrator
│   ├── preprocess.py           Load → validate → scale
│   ├── train.py                Train RF + GB, log to MLflow
│   └── evaluate.py             Metrics, drift detection, report
│
├── api/
│   └── app.py                  Flask REST API
│
├── models/                     Saved .pkl files + evaluation_report.json
│
├── tests/
│   └── test_pipeline.py        Pytest unit tests
│
├── mlflow.db                   SQLite MLflow tracking store (auto-created)
├── mlruns/                     MLflow artifact store (auto-created)
│
└── .kiro/
    └── steering/               Kiro AI context files
        ├── project.md
        ├── conventions.md
        └── tasks.md
```

---

## MLflow Integration

This project uses **MLflow with a SQLite backend** — no separate tracking server needed.

| What | Value |
|---|---|
| Tracking URI | `sqlite:///mlflow.db` |
| Artifact store | `./mlruns` |
| Experiment | `CardioSense_AI` |
| Run structure | Parent run → child runs per model |

### What gets logged

**Per model (child run):**
- All hyperparameters
- Accuracy, Precision, Recall, F1, AUC-ROC, Sensitivity, Specificity
- Confusion matrix counts (TP, TN, FP, FN)
- Classification report (artifact)
- Feature importance JSON (artifact)
- Registered model in MLflow Model Registry

**Parent run:**
- Best model name + metrics
- Production bundle artifact
- Drift detection results

### Launch the UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

---

## REST API

Start the server:
```bash
python api/app.py        # dev
gunicorn -w 2 -b 0.0.0.0:5000 "api.app:app"   # prod
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/model-info` | Model metadata + feature importance |
| POST | `/predict` | Single patient prediction |
| POST | `/predict/batch` | Batch predictions (up to 100) |
| GET | `/metrics` | Runtime stats |

### Example prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 58, "sex": 1, "cp": 0, "trestbps": 140,
    "chol": 268, "fbs": 0, "restecg": 0, "thalach": 152,
    "exang": 1, "oldpeak": 2.1, "slope": 1, "ca": 1, "thal": 2
  }'
```

Response includes: `prediction`, `risk_probability`, `risk_level` (LOW/MODERATE/HIGH), `recommendation`, `feature_contributions`, `latency_ms`.

---

## Tests

```bash
pytest tests/ -v --tb=short
```

Coverage:
- Data validation (missing values, duplicates, out-of-range)
- Preprocessing (shapes, column names, scaler zero-mean)
- Model evaluation (metric ranges, confusion matrix integrity)
- Drift detection (no-drift case, large-shift detection)
- API input validation (missing features, out-of-range values)

---

## Docker

```bash
# Build
docker build -t cardiosense:latest .

# Run prediction API
docker run -p 5000:5000 cardiosense:latest

# Run pipeline inside container
docker run cardiosense:latest python run_pipeline.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/heart.csv` | Input dataset |
| `MODELS_DIR` | `models/` | Where to save .pkl files |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
| `EXPERIMENT` | `CardioSense_AI` | MLflow experiment name |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `RANDOM_STATE` | `42` | Random seed |
| `BUNDLE_PATH` | `models/production_bundle.pkl` | API model bundle |
| `PORT` | `5000` | Flask port |

---

## Models & Results

| Model | Accuracy | AUC-ROC | F1 | Sensitivity | Specificity | Threshold |
|---|---|---|---|---|---|---|
| **RandomForest** ✅ | 77.1% | 0.882 | 0.794 | 0.818 | 0.714 | 0.60 |
| GradientBoosting | 85.2% | 0.869 | 0.873 | — | — | 0.60 |

**Production-readiness thresholds:** Sensitivity ≥ 0.80 ✅ · Specificity ≥ 0.70 ✅ · AUC ≥ 0.85 ✅

> Decision threshold tuned to **0.60** (vs default 0.50) to satisfy both clinical sensitivity and specificity constraints simultaneously.

---

## Authors

Arwin Dominic · Dinesh Kumar · Chethan K Chavan · Amaresh
