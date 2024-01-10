#!/usr/bin/env python
import argparse
import logging
import os
import pathlib
import tempfile

import mlflow
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    logger.info(f"Downloading {args.file_url} ...")

    # Create a temporary file in the specified directory with the custom name
    temp_file_path = os.path.join(tempfile.gettempdir(), args.artifact_name)

    logger.info("Creating download data run")
    with mlflow.start_run(description=args.artifact_description) as run:
        with requests.get(args.file_url, stream=True) as r:
            with open(temp_file_path, mode="wb") as fp:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)

        logger.info("Logging artifact")
        # Log the artifact using the custom name
        mlflow.log_artifact(temp_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to MLflow",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--file_url", type=str, help="URL to the input file", required=True
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
