import numpy as np
from typing import Tuple, Union


def runge_kutta2_step(f, _input, timestep, activity_map):
    X1 = f(activity_map, _input)
    return activity_map + timestep/2 * (X1 + f(activity_map + (timestep*X1), _input))


def extract_window(activity_map: np.ndarray, x: int, y: int, support: int):
    if x >= activity_map.shape[0]:
        raise IndexError(f"Cannot index x={x} in array of shape {activity_map.shape}")
    if y >= activity_map.shape[1]:
        raise IndexError(f"Cannot index y={y} in array of shape {activity_map.shape}")
    support = support // 2
    # add columns (or rows) if too near the edge
    if y < support:
        activity_map = np.concatenate((np.repeat(activity_map[:, 0][:, np.newaxis], support - y, axis=1), activity_map),
                                      axis=1)
        y = support

    elif y >= activity_map.shape[1] - support:
        activity_map = np.concatenate(
            (activity_map,
             np.repeat(activity_map[:, -1][:, np.newaxis], y + support - activity_map.shape[1] + 1, axis=1)),
            axis=1)

    if x < support:
        activity_map = np.concatenate((np.repeat(activity_map[0, :][np.newaxis, :], support - x, axis=0), activity_map),
                                      axis=0)
        x = support

    elif x >= activity_map.shape[0] - support:
        activity_map = np.concatenate(
            (activity_map,
             np.repeat(activity_map[-1, :][np.newaxis, :], x + support - activity_map.shape[0] + 1, axis=0)),
            axis=0)

    window = activity_map[x - support: x + support + 1, y - support: y + support + 1]
    return window


def gaussian_1d(x: Union[int, np.ndarray], mu: float, sigma: float, normalize: bool = False):
    hg = np.exp(-((x-mu) ** 2) / (2 * sigma ** 2))
    if normalize:
        return hg / np.sum(hg)
    else:
        return hg


def gaussian_2d(x: np.ndarray, y: np.ndarray, sigma: float, mu: Tuple[float]):
    hg = np.exp(-((x-mu[0]) ** 2 + (y-mu[1]) ** 2) / (2 * sigma ** 2))
    return hg / np.sum(hg)


def gaussian_kernel(support: int, sigma: float, mu: Tuple = (0, 0)):
    space_1D = np.linspace(-(support//2), (support//2), support)
    X, Y = np.meshgrid(space_1D, space_1D)
    return gaussian_2d(X, Y, sigma=sigma, mu=mu)
