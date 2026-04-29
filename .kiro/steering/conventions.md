# Coding Conventions

## Language & Runtime
- Python 3.12
- All ML code uses scikit-learn ≥ 1.4, pandas ≥ 2.2, numpy ≥ 1.26
- MLflow ≥ 3.0 with **SQLite backend** (`sqlite:///mlflow.db`)

## MLflow Conventions
- Always set tracking URI: `mlflow.set_tracking_uri("sqlite:///mlflow.db")`
- Experiment name: `CardioSense_AI`
- Use nested runs: parent run = full pipeline, child runs = per model
- Log all hyperparams with `mlflow.log_params(cfg["params"])`
- Log all metrics with `mlflow.log_metrics(metrics_dict)`
- Register models as `CardioSense_<ModelName>`
- Artifacts go in `./mlruns` relative to project root

## File Paths
- All paths resolve relative to **project root**, not `src/`
- Use `os.path` for cross-platform compatibility
- Data: `data/heart.csv`
- Models: `models/`
- MLflow DB: `mlflow.db` (project root)

## Code Style
- Module-level docstrings on every file
- Functions have docstrings
- Use `logging` not `print` — logger name = module name
- Type hints on public function signatures
- Return dicts for metrics, not tuples

## Testing
- Test file: `tests/test_pipeline.py`
- Run with: `pytest tests/ -v --tb=short`
- Tests must not require a live MLflow server (mock or disable active_run checks)
- Fixtures use `np.random.seed(42)` for reproducibility

## Environment Variables
| Variable | Default | Purpose |
|---|---|---|
| `DATA_PATH` | `data/heart.csv` | Input CSV |
| `MODELS_DIR` | `models` | Output directory for .pkl files |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow backend |
| `EXPERIMENT` | `CardioSense_AI` | MLflow experiment name |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `RANDOM_STATE` | `42` | Reproducibility seed |
| `PORT` | `5000` | Flask API port |
| `BUNDLE_PATH` | `models/production_bundle.pkl` | API model bundle |
