import numpy as np
import pandas as pd
from utils import runge_kutta2_step, gaussian_kernel
from typing import List, Tuple
from scipy import ndimage


class FEF:
    """
    Class Representing one map (with some feature preference) in the input preprocessor stage of the model
    This stage is attempting to model the dynamics of the path from the LGN to cortex which ``consists of a strong
    transient when the stimulus appears, and this transient is followed by a weaker sustained response.''
    The dynamics are represented in the update_v and update_w equations
    V represents faster excitatory cells
    W represents slower inhibitory cells
    """

    def __init__(self, parameters: pd.Series, feature_prefs: List[float], input_dim: Tuple[int, int]):
        self.feature_prefs = feature_prefs
        self.input_dim = (*input_dim, len(self.feature_prefs))
        self.V = np.zeros(shape=self.input_dim, dtype=np.double)
        self.leak_parameter = parameters["g1"]
        self.saturation = parameters["e1"]
        self.noise_sigma = parameters["noise_sigma"]
        self.beta = parameters["beta"]
        self.time_delay = 1 / parameters["c1"]

        self.kernel = np.zeros((int(parameters["plus_supp"]), int(parameters["plus_supp"]), 1))
        self.kernel[:, :, 0] = gaussian_kernel(int(parameters["plus_supp"]),
                                               parameters["sigma_plus"])

    def V_dot(self, V: np.ndarray, feedforward_signal: np.ndarray) -> np.ndarray:
        """
        :param V:
        :param feedforward_signal:
        :return:
        """

        leak_conductance = - self.leak_parameter * V
        lateral_inhibition = - self.beta * ndimage.convolve(V, self.kernel, mode='nearest')
        excitatory = feedforward_signal
        driving = leak_conductance + lateral_inhibition + excitatory

        noise = np.random.normal(0.0, self.noise_sigma, self.input_dim)

        v_dot = driving + noise
        return self.time_delay * v_dot

    def update(self, feedforward_signal: np.ndarray, timestep: float) -> None:
        """
        Update excitatory (V) cell
        """
        self.V = runge_kutta2_step(self.V_dot, feedforward_signal, timestep, self.V)
