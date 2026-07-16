import os
import json
import pickle
import yaml
import numpy as np
from pathlib import Path
from PIL import Image
import nibabel as nib


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def load_pickle(file: str, mode: str = "rb"):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def load_yaml(file: str):
    with open(file, "r") as f:
        a = yaml.load(f, Loader=yaml.BaseLoader)
    return a


def read_file_to_nifti_or_np(imagepath, dtype=np.float32):
    path = Path(imagepath)
    suffixes = [s.lower() for s in path.suffixes]

    if suffixes[-2:] == [".nii", ".gz"] or suffixes[-1:] == [".nii"]:
        return nib.load(str(path))

    ext = suffixes[-1] if suffixes else ""

    if ext in {".png", ".jpg", ".jpeg"}:
        return np.array(Image.open(path).convert("L"), dtype=dtype)

    if ext in {".csv", ".txt"}:
        return np.atleast_1d(np.genfromtxt(path, delimiter=",", dtype=dtype))

    raise TypeError(
        f"File type invalid. Found extension: {''.join(suffixes) or '<none>'} "
        "and expected one in [.nii, .nii.gz, .png, .jpg, .jpeg, .csv, .txt]"
    )
