import numpy as np
from typing import Tuple


def runge_kutta2_step(f, _input, timestep, activity_map):
    X1 = f(activity_map, _input)
    return activity_map + timestep / 2 * (X1 + f(activity_map + (timestep * X1), _input))


def gaussian_2d(x: np.ndarray, y: np.ndarray, sigma: float, mu: Tuple[float]):
    hg = np.exp(-((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / (2 * sigma ** 2))
    return hg / np.sum(hg)


def gaussian_kernel(support: int, sigma: float, mu: Tuple = (0, 0)):
    space_1D = np.linspace(-(support // 2), (support // 2), support)
    X, Y = np.meshgrid(space_1D, space_1D)
    return gaussian_2d(X, Y, sigma=sigma, mu=mu)

