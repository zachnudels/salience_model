import numpy as np


class InputProcessor:
    """
    Class Representing one map (with some feature preference) in the input preprocessor stage of the model
    This stage is attempting to model the dynamics of the path from the LGN to cortex which ``consists of a strong
    transient when the stimulus appears, and this transient is followed by a weaker sustained response.''
    The dynamics are represented in the update_v and update_w equations
    V represents faster excitatory cells
    W represents slower inhibitory cells
    """
    def __init__(self, feature_pref: int, input_dim: int):
        self.feature_pref = feature_pref
        self.V = np.zeros(shape=(input_dim, input_dim))
        self.W = np.zeros(shape=(input_dim, input_dim))

    def V_dot(self, signal: np.ndarray) -> np.ndarray:
        activity = np.ones_like(signal) * (signal == self.feature_pref)  # 1 if feature is preferred by this map

        excitatory = activity * (10 - self.V)
        inhibitory = 2 * self.W ** 2 * self.V
        v_dot = excitatory - inhibitory
        return v_dot

    def W_dot(self) -> np.ndarray:
        excitatory = self.V * (25 - self.W)
        inhibitory = self.W
        return (excitatory - inhibitory) / 5

    def update(self, _input: np.ndarray, timestep: float) -> None:
        V_dot = self.V_dot(_input)
        W_dot = self.W_dot()

        self.V += V_dot * timestep
        self.W += W_dot * timestep

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"V:\n{self.V}, \n W:\n{self.W}\n"
