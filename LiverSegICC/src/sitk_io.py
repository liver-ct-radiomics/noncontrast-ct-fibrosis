"""NIfTI reading and writing that survives non-ASCII paths.

On Windows, ITK's NIfTI library cannot open a path containing characters such as
"Ç" or "ş" (for example a user name). When the path is not plain ASCII the file is
opened from inside its own folder with a relative, ASCII-only file name. Every file
name this program creates is ASCII, so only the folder part can cause the problem.
The working directory is process-wide, hence the lock.
"""

import os
import threading
from pathlib import Path

import SimpleITK as sitk

_lock = threading.RLock()


def read_image(path):
    path = Path(path)
    if str(path).isascii():
        return sitk.ReadImage(str(path))
    with _lock:
        previous = os.getcwd()
        os.chdir(path.parent)
        try:
            return sitk.ReadImage(path.name)
        finally:
            os.chdir(previous)


def write_image(image, path, compress = True):
    path = Path(path)
    if str(path).isascii():
        sitk.WriteImage(image, str(path), compress)
        return
    with _lock:
        previous = os.getcwd()
        os.chdir(path.parent)
        try:
            sitk.WriteImage(image, path.name, compress)
        finally:
            os.chdir(previous)
