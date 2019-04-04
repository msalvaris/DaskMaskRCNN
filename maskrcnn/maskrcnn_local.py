import logging.config
import os
import fire
from maskrcnn import dask_pipeline


def run(scheduler_address, config_file, filepath, output_path):
    logging.config.fileConfig(os.getenv("LOG_CONFIG", "logging.ini"))

    dask_pipeline.start(config_file, filepath, output_path, scheduler_address)


if __name__ == "__main__":
    fire.Fire(run)
