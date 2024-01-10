#!/usr/bin/env python
import argparse
import logging
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import mlflow

from sklearn.manifold import TSNE

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    with mlflow.start_run(description=args.artifact_description):
        logger.info("Downloading artifact")
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=args.run_id, artifact_path=args.input_artifact
        )

        iris = pd.read_csv(
            artifact_path,
            skiprows=1,
            names=(
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
                "target",
            ),
        )

        target_names = "setosa,versicolor,virginica".split(",")
        iris["target"] = [target_names[k] for k in iris["target"]]

        logger.info("Performing t-SNE")
        tsne = TSNE(n_components=2, init="pca", random_state=0)
        transf = tsne.fit_transform(iris.iloc[:, :4])

        iris["tsne_1"] = transf[:, 0]
        iris["tsne_2"] = transf[:, 1]

        g = sns.displot(iris, x="tsne_1", y="tsne_2", hue="target", kind="kde")

        # Save the plot as an image
        image_path = "tsne_plot.png"
        g.fig.savefig(image_path)
        plt.close()  # Close the plot to free up resources

        logger.info("Uploading image to MLflow")
        mlflow.log_artifact(image_path)

        logger.info("Creating artifact")

        iris.to_csv("clean_data.csv")

        logger.info("Logging artifact")
        mlflow.log_artifact(args.artifact_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process data and upload it as an artifact to MLflow",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--run_id",
        type=str,
        help="Previous Run ID",
        required=True,
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
