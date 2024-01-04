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
    with tempfile.NamedTemporaryFile(mode="wb+") as fp:
        logger.info("Creating run demo1")
        with mlflow.start_run() as run:
            with requests.get(args.file_url, stream=True) as r:
                for chunk in r.iter_content(chunk_size=8192):
                    fp.write(chunk)
            fp.flush()

            logger.info("Logging artifact")
            mlflow.log_artifact(fp.name, args.artifact_name)


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
