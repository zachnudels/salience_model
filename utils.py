import math

import numpy as np

from scipy import ndimage
from functools import partial


def runge_kutta2_step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + k1, y + h)

    # Update next value of y
    return x + (k1 + k2) / 2


def extract_window(activity_map: np.ndarray, x: int, y: int, support: int):
    support = support // 2

    # add columns (or rows) if on the edge
    if x < support:
        x = support
        activity_map = np.concatenate((np.repeat(activity_map[:, 0][:, np.newaxis], support - x, axis=1), activity_map),
                                      axis=1)

    elif x > activity_map.shape[0] - support:
        x = activity_map.shape[0] - support
        activity_map = np.concatenate(
            (activity_map, np.repeat(activity_map[:, -1][:, np.newaxis], activity_map.shape[0] - support + x, axis=1)),
            axis=1)

    if y < support:
        y = support
        activity_map = np.concatenate((np.repeat(activity_map[0, :][np.newaxis, :], support - y, axis=0), activity_map),
                                      axis=0)

    elif y > activity_map.shape[1] - support:
        y = activity_map.shape[1] - support
        activity_map = np.concatenate(
            (activity_map, np.repeat(activity_map[-1, :][np.newaxis, :], activity_map.shape[0] - support + y, axis=0)),
            axis=0)

    window = activity_map[x - support: x + support + 1, y - support: y + support + 1]
    return window


def receptive_field_activity(x: int, y: int, support: float, sigma: float, activity_map: np.ndarray):
    window = extract_window(activity_map, x, y, int(support))
    return ndimage.gaussian_filter(window, sigma)[support // 2, support // 2]


def feedforward_signal(lower_activity_map: np.ndarray,
                       support: float,
                       sigma: float,
                       X: np.ndarray,
                       Y: np.ndarray) -> np.ndarray:
    vectorized_rf_activity = np.vectorize(
        partial(receptive_field_activity, activity_map=lower_activity_map, support=int(support), sigma=sigma))

    return vectorized_rf_activity(X, Y)


def feedback_signal(higher_activity_map: np.ndarray,
                    input_dim_h: int,
                    input_dim_l: int,
                    support: float,
                    sigma: float,
                    X: np.ndarray,
                    Y: np.ndarray) -> np.ndarray:
    vectorized_rf_activity = np.vectorize(
        partial(receptive_field_activity, activity_map=higher_activity_map, support=int(support), sigma=sigma))
    rfs = vectorized_rf_activity(X, Y)

    repeats = math.ceil(input_dim_l / input_dim_h)
    rfs = np.repeat(rfs, repeats, axis=0)
    rfs = np.repeat(rfs, repeats, axis=1)

    return rfs
