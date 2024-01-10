import mlflow
import os
import hydra
from omegaconf import DictConfig


# This automatically reads in the configuration
@hydra.main(config_name="config")
def go(config: DictConfig):
    # Setup the mlflow experiment. All runs will be grouped under this name
    os.environ["MLFLOW_TRACKING_URI"] = config["main"]["mlflow_tracking_uri"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    download_data = mlflow.run(
        uri=os.path.join(root_path, "download_data"),
        entry_point="main",
        experiment_name=config["main"]["experiment_name"],
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "iris.csv",
            "artifact_description": "Input data",
        },
    )

    process_data = mlflow.run(
        uri=os.path.join(root_path, "process_data"),
        entry_point="main",
        experiment_name=config["main"]["experiment_name"],
        parameters={
            "run_id": download_data.run_id,
            "input_artifact": "iris.csv",
            "artifact_name": "clean_data.csv",
            "artifact_description": "Cleaned data",
        },
    )


if __name__ == "__main__":
    go()
