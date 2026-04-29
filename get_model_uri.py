"""
get_model_uri.py
================
Prints the MLflow model URI and serving command for the latest
registered CardioSense_RandomForest model.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import mlflow
from mlflow.tracking import MlflowClient

ROOT = os.path.dirname(os.path.abspath(__file__))
MLFLOW_URI = f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()

model_name = "CardioSense_RandomForest"

# Get latest version
versions = client.get_latest_versions(model_name)
if not versions:
    print(f"No versions found for model: {model_name}")
    sys.exit(1)

latest = sorted(versions, key=lambda v: int(v.version))[-1]
run_id = latest.run_id
version = latest.version

# Build URIs
registry_uri = f"models:/{model_name}/{version}"
run_uri      = f"runs:/{run_id}/model"

print(f"\n{'='*60}")
print(f"  Model:    {model_name}")
print(f"  Version:  {version}")
print(f"  Run ID:   {run_id}")
print(f"  Status:   {latest.current_stage}")
print(f"{'='*60}")
print(f"\nRegistry URI:  {registry_uri}")
print(f"Run URI:       {run_uri}")
print(f"\n{'='*60}")
print("  SERVING COMMANDS")
print(f"{'='*60}")
print(f"\n1. Serve from Model Registry (recommended):")
print(f"   mlflow models serve -m \"{registry_uri}\" --port 5002 --no-conda")
print(f"\n2. Serve from Run artifacts:")
print(f"   mlflow models serve -m \"{run_uri}\" --port 5002 --no-conda")
print(f"\n3. Serve with custom host (accessible on network):")
print(f"   mlflow models serve -m \"{registry_uri}\" --host 0.0.0.0 --port 5002 --no-conda")
print(f"\nTracking URI: {MLFLOW_URI}")
print(f"\nTest the deployed model:")
print("""   curl -X POST http://127.0.0.1:5002/invocations \\
     -H "Content-Type: application/json" \\
     -d '{"dataframe_records": [{"age":58,"sex":1,"cp":0,"trestbps":140,
          "chol":268,"fbs":0,"restecg":0,"thalach":152,"exang":1,
          "oldpeak":2.1,"slope":1,"ca":1,"thal":2}]}'""")
