import numpy as np
import pandas as pd
from utils import runge_kutta2_step, gaussian_1d
from typing import Tuple

class LGN:
    """
    Class Representing one map (with some feature preference) in the input preprocessor stage of the model
    This stage is attempting to model the dynamics of the path from the LGN to cortex which ``consists of a strong
    transient when the stimulus appears, and this transient is followed by a weaker sustained response.''
    The dynamics are represented in the update_v and update_w equations
    V represents faster excitatory cells
    W represents slower inhibitory cells
    """

    def __init__(self, parameters: pd.Series, feature_pref: float, input_dim: Tuple[int, int]):
        self.feature_pref = feature_pref
        self.V = np.zeros(shape=input_dim, dtype=np.double)
        self.W = np.zeros(shape=input_dim, dtype=np.double)
        self.rf_width = parameters["rf_width"]

    def V_dot(self, V: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        :param V:
        :param signal:
        :return:
        """
        activity = np.ones_like(signal) * (signal == self.feature_pref)  # 1 if feature is preferred by this map
        inhibitory = -2 * self.W ** 2 * V
        excitatory = -activity * (V - 10)
        v_dot = inhibitory + excitatory
        return v_dot

    def W_dot(self, W: np.ndarray, _input=None) -> np.ndarray:
        """
        :return:
        """
        inhibitory = - (1/5) * W
        excitatory = - (1/5) * self.V * (W - 25)
        return inhibitory + excitatory

    def update(self, _input: np.ndarray, timestep: float) -> None:
        """
        First update excitatory (V) cell using the previous inhibitory (W) cell
        Then update the inhibitory cell (W) using the new V cell
        """
        self.V = runge_kutta2_step(self.V_dot, _input, timestep, self.V)
        self.W = runge_kutta2_step(self.W_dot, _input, timestep, self.W)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"V:\n{self.V}, \n W:\n{self.W}\n"
