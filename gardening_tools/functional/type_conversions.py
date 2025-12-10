import numpy as np
import nibabel as nib
from typing import Union


def nifti_or_np_to_np(array: Union[np.ndarray, nib.Nifti1Image]) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, (nib.Nifti1Image, nib.minc1.Minc1Image)):
        return array.get_fdata().astype(np.float32)
    else:
        raise TypeError(
            f"File data type invalid. Found: {type(array)} and expected nib.Nifti1Image or np.ndarray"
        )
