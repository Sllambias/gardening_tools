import numpy as np


def verify_labels_are_equal(
    expected_labels: np.ndarray, actual_labels: np.ndarray, id=""
):
    if expected_labels.dtype == actual_labels.dtype:
        if np.all(np.isin(actual_labels, expected_labels)):
            return True
        else:
            print(
                f"Unexpected labels found for {id} \n"
                f"expected: {expected_labels} \n"
                f"found: {actual_labels}"
            )
            return False
    else:
        print(
            "make sure reference and target is the same dtype before comparing the labels. \n"
            f"reference is: {expected_labels.dtype} and target is: {actual_labels.dtype}"
        )
        return False


def verify_shapes_are_equal(reference_shape, target_shape, id=""):
    if np.all(reference_shape == target_shape):
        return True
    else:
        print(
            f"Sizes do not match for {id}"
            f"Image is: {reference_shape} while the label is {target_shape}"
        )
        return False
