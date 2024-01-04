# Demo 2 - Versioning Data and Artifacts

Open Jupyter Notebook - noteboook/population.ipynb

```
import mlflow
mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.create_experiment("demo-2")
mlflow.set_experiment("demo-2")

# Uploading the Artifact
with mlflow.start_run():
    mlflow.log_artifact("/workspaces/codespaces-jupyter/data/atlantis.csv", artifact_path="data")

# Uploading a New Version
with mlflow.start_run():
    mlflow.log_artifact("/workspaces/codespaces-jupyter/data/atlantis.csv", artifact_path="data")
```