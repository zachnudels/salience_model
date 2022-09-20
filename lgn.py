import numpy as np
from typing import List, Tuple

from utils import runge_kutta2_step


class LGN:
    """
    Class Representing one map (with some feature preference) in the input preprocessor stage of the model
    This stage is attempting to model the dynamics of the path from the LGN to cortex which ``consists of a strong
    transient when the stimulus appears, and this transient is followed by a weaker sustained response.''
    The dynamics are represented in the update_v and update_w equations
    V represents faster excitatory cells
    W represents slower inhibitory cells
    """

    def __init__(self, feature_prefs: List[float], input_dim: Tuple[int, int]):
        self.feature_prefs = feature_prefs
        self.input_dim = (*input_dim, len(self.feature_prefs))
        self.V = np.zeros(shape=self.input_dim, dtype=np.double)
        self.W = np.zeros(shape=self.input_dim, dtype=np.double)

    def V_dot(self, V: np.ndarray, signal: np.ndarray) -> np.ndarray:
        """
        :param V:
        :param signal:
        :return:
        """
        inhibitory = -2 * self.W ** 2 * V
        excitatory = -signal * (V - 10)
        v_dot = inhibitory + excitatory
        return v_dot

    def W_dot(self, W: np.ndarray, _input=None) -> np.ndarray:
        """
        :return:
        """
        inhibitory = - W
        excitatory = - self.V * (W - 25)
        return (1/5) * (inhibitory + excitatory)

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
