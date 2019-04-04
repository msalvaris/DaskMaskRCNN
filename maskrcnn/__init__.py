import glob
from timeit import default_timer

import numpy as np
from PIL import Image
from toolz import curry, pipe
import logging

logger = logging.getLogger(__name__)


def save_image(filename, data):
    img= data[:,:,[2,1,0]]
    # img = data.clone().clamp(0, 255).numpy()
    # img = img.transpose(1, 2, 0).astype("uint8")
    img=img.astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


@curry
def resize(pil_img, size, interpolation=Image.BILINEAR):
    return pil_img.resize(size, interpolation)


def hwc_to_chw(img_array):
    return np.moveaxis(img_array, 2, 0)

def _convert_to_bgr(img_array):
    return img_array[:,:, [2, 1, 0]]


class FileReader(object):
    def __init__(self, path, recursive=True):
        self._path = path
        self._recursive = recursive
        self._previous_files = {}

    def _list_files(self):
        return glob.glob(self._path, recursive=self._recursive)

    def new_files(self):
        current_files = set(self._list_files())
        new_files = current_files.difference(self._previous_files)
        if len(new_files) > 0:
            logger.info("Found {} new files".format(len(new_files)))
        self._previous_files = current_files
        return new_files


class CountdownTimer(object):
    def __init__(self, duration=60):
        self._duration = duration
        self._timer = default_timer
        self.reset()

    def reset(self, duration=None):
        self._duration = self._duration if duration is None else duration
        self._start = self._timer()

    def is_expired(self):
        if self._timer() - self._start > self._duration:
            return True
        else:
            return False


def load_image(filepath):
    return pipe(filepath, pil_loader, np.array, _convert_to_bgr)
