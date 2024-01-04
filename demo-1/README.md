# Demo 1 - Deploying MLflow to Localhost

```
conda init
source .bashrc
conda env list
conda create -n mlflow python=3.9
conda activate mlflow
pip install mlflow
mkdir mlflow-server
cd mlflow-server
mlflow ui
```

Open Mlflow port with Web Browser.

Open Jupyter Notebook - noteboook/population.ipynb

```
import mlflow
mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.create_experiment("demo-1")
mlflow.set_experiment("demo-1")

# Initial Run
with mlflow.start_run():
    print("Hello, World!)
```