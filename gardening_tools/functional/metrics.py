import numpy as np


def dice(tp, fp, tn, fn):
    try:
        return (2 * tp) / (2 * tp + fp + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def dice_per_label(tp_list, fp_list, tn_list, fn_list):
    return [
        dice(tp_list[i], fp_list[i], tn_list[i], fn_list[i])
        for i in range(len(tp_list))
    ]


def jaccard(tp, fp, tn, fn):
    try:
        return (tp) / (tp + fp + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def jaccard_per_label(tp_list, fp_list, tn_list, fn_list):
    return [
        jaccard(tp_list[i], fp_list[i], tn_list[i], fn_list[i])
        for i in range(len(tp_list))
    ]


def sensitivity(tp, fp, tn, fn):
    try:
        return tp / (tp + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def specificity(tp, fp, tn, fn):
    try:
        return tn / (tn + fp)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def precision(tp, fp, tn, fn):
    try:
        return tp / (tp + fp)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def volume_similarity(tp, fp, tn, fn):
    try:
        return 1 - abs(fn - fp) / (2 * tp + fn + fp)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def f1(tp, fp, tn, fn):
    try:
        prec = precision(tp, fp, tn, fn)
        sens = sensitivity(tp, fp, tn, fn)
        return 2 * (prec * sens) / (prec + sens)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def accuracy(tp, fp, tn, fn):
    try:
        return (tp + tn) / (tp + fp + tn + fn)
    except (ZeroDivisionError, RuntimeWarning):
        if tp + fn > 0:
            return 0
        else:
            return np.nan


def TP(tp, fp, tn, fn):
    return tp


def FP(tp, fp, tn, fn):
    return fp


def FN(tp, fp, tn, fn):
    return fn


def total_pos_gt(tp, fp, tn, fn):
    return tp + fn


def total_pos_pred(tp, fp, tn, fn):
    return tp + fp
