import numpy as np
import pandas as pd

from scipy import ndimage
from typing import Callable, List, Tuple
from utils import gaussian_kernel, runge_kutta2_step


class Layer:
    """
    Class Representing region unit (with some feature preference) i.e., V1, V2, V4 of the model
    Each unit has four cells (represented as floats) interacting with one another as well as other units
    in higher and lower regions
    """

    def __init__(self,
                 parameters: pd.Series,
                 input_dim: Tuple[int, int],
                 feature_prefs: List[float],
                 tuning_curve: Callable):

        self.tuning_curve = tuning_curve
        self.feature_prefs = feature_prefs
        self.input_dim = (*input_dim, len(self.feature_prefs))

        # CELLS
        self.V = np.zeros(shape=self.input_dim, dtype=np.double)  # Input cell
        self.D = np.zeros(shape=self.input_dim, dtype=np.double)  # Feedback cell (region filling)
        self.U = np.zeros(shape=self.input_dim, dtype=np.double)  # Inhibitory cell
        self.W = np.zeros(shape=self.input_dim, dtype=np.double)  # Edge detection cell
        self.S = np.zeros(shape=self.input_dim, dtype=np.double)  # Similarity cell

        # PARAMETERS
        self.a = parameters["a"]  # V input weight
        self.k1 = parameters["k1"]  # V Feedback effect
        self.k2 = parameters["k2"]  # V Similarity effect
        self.c1 = parameters["c1"]  # V time scale
        self.c2 = parameters["c2"]  # W time scale
        self.c3 = parameters["c3"]  # U time scale
        self.c4 = parameters["c4"]  # D time scale
        self.e1 = parameters["e1"]  # V driving input threshold
        self.e2 = parameters["e2"]  # V edge detection threshold
        self.e3 = parameters["e3"]  # W edge detection threshold
        self.e4 = parameters["e4"]  # U excitatory threshold
        self.e5 = parameters["e5"]  # D excitatory threshold
        self.e6 = parameters["e6"]  # S excitatory threshold
        self.g1 = parameters["g1"]  # V leak conductance
        self.g2 = parameters["g2"]  # V edge detection modulation
        self.g3 = parameters["g3"]  # W leak conductance
        self.g4 = parameters["g4"]  # W Boundary detection
        self.g5 = parameters["g5"]  # U leak conductance
        self.g6 = parameters["g6"]  # U excitatory weight
        self.g7 = parameters["g7"]  # D leak conductance
        self.g8 = parameters["g8"]  # S excitatory modulation
        self.g9 = parameters["g9"]  # S similarity modulation

        # KERNELS
        self.ff_kernel = gaussian_kernel(int(parameters["ff_support"]), parameters["sigma_ff"])
        self.fb_kernel = gaussian_kernel(int(parameters["fb_support"]), parameters["sigma_fb"])

        self.plus_kernel = np.zeros((int(parameters["plus_supp"]), int(parameters["plus_supp"]), 1))
        self.plus_kernel[:, :, 0] = gaussian_kernel(int(parameters["plus_supp"]),
                                                    parameters["sigma_plus"])

        self.minus_kernel = np.zeros((int(parameters["minus_supp"]), int(parameters["minus_supp"]), 1))
        self.minus_kernel[:, :, 0] = gaussian_kernel(int(parameters["minus_supp"]),
                                                     parameters["sigma_minus"])

    def V_dot(self, V: np.ndarray, feedforward_signal: np.ndarray) -> np.ndarray:
        """

        :param V:
        :param feedforward_signal:
        :return:
        """
        leak_conductance = - self.g1 * self.U

        if not pd.isna(self.k2):
            leak_conductance -= self.k2 * self.S

        leak_conductance *= V

        feedback_modulation = 1
        # There is no effect of D cell in V4 so do not compute for that case
        if not pd.isna(self.k1):
            feedback_modulation += self.k1 * self.D

        driving_input = - self.a * feedforward_signal * feedback_modulation * (V - self.e1)
        boundary_detection = - self.g2 * self.W * (V - self.e2)

        V_dot = leak_conductance + driving_input + boundary_detection

        return self.c1 * V_dot

    def W_dot(self, W, _input):
        inhibitory = - self.g3 * W
        if not pd.isna(self.k1):
            inhibitory *= self.D

        V = - self.g4 * np.abs(_input - ndimage.correlate(
            _input, self.minus_kernel, mode='nearest'
        )) * (W - self.e3)

        return (1 / self.c2) * (V + inhibitory)

    def U_dot(self, U: np.ndarray, _input=None) -> np.ndarray:
        """
        :return:
        """
        leak_conductance = - self.g5 * U
        excitatory = - self.g6 * self.V * (U - self.e4)

        U_dot = leak_conductance + excitatory
        return (1 / self.c3) * U_dot

    def S_dot(self, S: np.ndarray, _input) -> np.ndarray:
        excitatory = np.zeros_like(S)
        for k1 in range(len(self.feature_prefs)):
            for k2 in range(len(self.feature_prefs)):
                if k1 != k2:
                    excitatory[:, :, k1] += \
                        self.g9 * self.tuning_curve(self.feature_prefs[k1], self.feature_prefs[k2]) *\
                        _input[:, :, k2] * _input[:, :, k1]
        excitatory = -self.g8 * excitatory * (self.S - self.e6)
        return excitatory

    def D_dot(self, D: np.ndarray, feedback_signal: np.ndarray) -> np.ndarray:
        """
        :param D:
        :param feedback_signal:
        :return:
        """
        leak_conductance = - self.g7 * D
        feedback_effect = - feedback_signal * (D - self.e5)

        return (1 / self.c4) * (leak_conductance + feedback_effect)

    def update(self,
               feedforward_signal: np.ndarray,
               timestep: float) -> None:
        """
        """
        if not pd.isna(self.k2):
            self.S = runge_kutta2_step(self.S_dot, self.V, timestep, self.S)
        self.V = runge_kutta2_step(self.V_dot, feedforward_signal, timestep, self.V)
        if not pd.isna(self.c3):
            self.U = runge_kutta2_step(self.U_dot, None, timestep, self.U)
        self.W = runge_kutta2_step(self.W_dot, self.V, timestep, self.W)

    def update_D(self, feedback_signal: np.ndarray, timestep: float) -> None:
        if not pd.isna(self.g2):  # V4 cell - for efficiency don't compute since will be 0
            self.D = runge_kutta2_step(self.D_dot, feedback_signal, timestep, self.D)


def f(x):
    return (np.abs(x) + x) / 2
