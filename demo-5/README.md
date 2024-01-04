# Demo 5 - MLflow Project Component

export MLFLOW_TRACKING_URI="http://localhost:5000"
export MLFLOW_EXPERIMENT_NAME="demo-5"

mlflow run . -P file_url=https://raw.githubusercontent.com/scikit-learn/scikit-learn/4dfdfb4e1bb3719628753a4ece995a1b2fa5312a/sklearn/datasets/data/iris.csv -P artifact_name=iris -P artifact_description="The sklearn IRIS dataset"
