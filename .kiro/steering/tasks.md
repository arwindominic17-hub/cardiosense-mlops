# Tasks & Workflows

## Daily Dev Tasks

### Run the full ML pipeline
```bash
python run_pipeline.py
```
This runs: data ingestion → validation → preprocessing → training → evaluation → drift detection → report generation.

### Open MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```
Then open http://localhost:5001 in your browser.

### Run unit tests
```bash
pytest tests/ -v --tb=short
```

### Start the prediction API
```bash
python api/app.py
# or with gunicorn:
gunicorn -w 2 -b 0.0.0.0:5000 "api.app:app"
```

### Test a prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":58,"sex":1,"cp":0,"trestbps":140,"chol":268,"fbs":0,"restecg":0,"thalach":152,"exang":1,"oldpeak":2.1,"slope":1,"ca":1,"thal":2}'
```

### Check API health
```bash
curl http://localhost:5000/health
```

## CI/CD (GitHub Actions)
Pipeline defined in `.github/workflows/mlops_pipeline.yml`.
Triggers on push to `main`. Runs: lint → test → train → build Docker image.

## Docker
```bash
# Build
docker build -t cardiosense:latest .

# Run API
docker run -p 5000:5000 cardiosense:latest

# Run pipeline inside container
docker run cardiosense:latest python run_pipeline.py
```

## Troubleshooting

### "Model not loaded" from API
Run `python run_pipeline.py` first to generate `models/production_bundle.pkl`.

### MLflow FutureWarning about filesystem store
This is expected if using old `file:./mlruns` URI. The code now uses `sqlite:///mlflow.db` which silences it.

### Tests failing with import errors
Make sure you're running from the project root:
```bash
cd cardiosense_mlops && pytest tests/ -v
```
