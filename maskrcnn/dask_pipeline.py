import logging
import os
from threading import Thread
from time import sleep
from timeit import default_timer

import dask
from dask.distributed import as_completed, Client
from maskrcnn_benchmark.config import cfg
from toolz import curry

from maskrcnn import CountdownTimer, FileReader, save_image, load_image
from maskrcnn.model import (
    score_batch,
    add_annotations,
    create_preprocessing,
    load_model,
    clean_gpu_mem,
)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


@curry
def write(output_folder, filename, img_array):
    filename = os.path.split(filename)[-1]
    outpath = os.path.join(output_folder, filename)
    save_image(outpath, img_array)
    return 1


def loop_annotations(orig_image_list, prediction_list):
    return [
        add_annotations(orig_image, prediction)
        for orig_image, prediction in zip(orig_image_list, prediction_list)
    ]


@curry
def loop_write(output_path, batch_list, results_list):
    for batch, results in zip(batch_list, results_list):
        write(output_path, batch, results)
    return True


@curry
def process_batch(client, style_model, preprocessing, output_path, batch):
    remote_batch_f = client.scatter(batch)
    img_array_f = client.map(load_image, remote_batch_f)
    pre_img_array_f = client.map(preprocessing, img_array_f)
    styled_array_f = client.submit(score_batch, style_model, pre_img_array_f)
    results_f = client.submit(loop_annotations, img_array_f, styled_array_f)
    return client.submit(loop_write(output_path), batch, results_f)


def score_images(
    processing_func, file_reader, batch_size=4, sleep_period=0.1, patience=60
):
    logger = logging.getLogger(__name__)
    patience_timer = CountdownTimer(duration=patience)
    all_res = []
    while True:
        new_files = file_reader.new_files()
        if len(new_files) > 0:
            patience_timer.reset()
            new_res = list(map(processing_func, chunks(list(new_files), batch_size)))
            all_res.extend(new_res)

        for res in all_res:
            if res.done():
                all_res.remove(res)
        logger.debug("Batches remaining {}".format(len(all_res)))

        if patience_timer.is_expired() and len(all_res) == 0:
            logger.info("Finished processing images")
            break

        sleep(sleep_period)


def _distribute_model_to_workers(client, config):
    logger = logging.getLogger(__name__)
    logger.info("Loading model...")
    start = default_timer()
    maskrcnn_model = client.submit(load_model, config)
    dask.distributed.wait(maskrcnn_model)
    client.replicate(maskrcnn_model)
    logger.info(
        "Model replicated on workers | took {} seconds".format(default_timer() - start)
    )
    return maskrcnn_model


def run_maskrcnn_pipeline(
    client, config_file, filepath, output_path, patience=60, batch_size=4
):
    logger = logging.getLogger(__name__)
    logger.info("Running Mask-RCNN")
    logger.info(f"Loading config {config_file}")
    cfg.merge_from_file(config_file)
    logger.debug(str(cfg))
    maskrcnn_model = _distribute_model_to_workers(client, cfg)

    filepath = os.path.join(filepath, "*.jpg")
    logger.info("Reading files from {}".format(filepath))
    file_reader = FileReader(filepath)

    logger.info("Writing files to {}".format(output_path))
    processing_func = process_batch(
        client, maskrcnn_model, create_preprocessing(cfg), output_path
    )

    load_thread = Thread(
        target=score_images,
        args=(processing_func, file_reader),
        kwargs={"patience": patience, "batch_size": batch_size},
    )
    start = default_timer()
    load_thread.start()
    load_thread.join()
    logger.info("Finished processing images in {}".format(default_timer() - start))

    # Delete model and clear GPU memory
    logger.info("Clearing model from GPU")
    del maskrcnn_model
    client.run(clean_gpu_mem)


@curry
def start(
    config_file, filepath, output_path, scheduler_address, patience=60, batch_size=4
):
    client = Client(scheduler_address)
    logger = logging.getLogger(__name__)
    logger.info(str(client))
    run_maskrcnn_pipeline(
        client,
        config_file,
        filepath,
        output_path,
        patience=patience,
        batch_size=batch_size,
    )
    client.close()
