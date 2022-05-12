import numpy as np


class LGN:
    """
    Class Representing one map (with some feature preference) in the input preprocessor stage of the model
    This stage is attempting to model the dynamics of the path from the LGN to cortex which ``consists of a strong
    transient when the stimulus appears, and this transient is followed by a weaker sustained response.''
    The dynamics are represented in the update_v and update_w equations
    V represents faster excitatory cells
    W represents slower inhibitory cells
    """

    def __init__(self, feature_pref: float, input_dim: int):
        self.feature_pref = feature_pref
        self.V = np.zeros(shape=(input_dim, input_dim))
        self.W = np.zeros(shape=(input_dim, input_dim))

    def V_dot(self, signal: np.ndarray) -> np.ndarray:
        # # If feature matches preferences, 1
        # # If orthogonal, 0
        # # If close, should be close to 1
        # # TODO: Make preference circular
        # activity = np.ones_like(signal) * (1 - np.abs(signal-self.feature_pref))

        activity = np.ones_like(signal) * (signal == self.feature_pref)  # 1 if feature is preferred by this map
        excitatory = activity * (10 - self.V)
        inhibitory = 2 * self.W ** 2 * self.V
        v_dot = excitatory - inhibitory
        return v_dot

    def W_dot(self) -> np.ndarray:
        excitatory = self.V * (25 - self.W)
        inhibitory = self.W
        divisor = 1 / 5
        return (excitatory - inhibitory) * divisor

    def update(self, _input: np.ndarray, timestep: float) -> None:
        """
        First update excitatory (V) cell using the previous inhibitory (W) cell
        Then update the inhibitory cell (W) using the new V cell
        areaLGNExc.W = areaLGNInh.V; areaLGNExc = updateNeuronField(areaLGNExc);
        areaLGNInh.Gin = areaLGNExc.V; areaLGNInh = updateNeuronField(areaLGNInh);
        """
        V_dot = self.V_dot(_input)
        self.V += V_dot * timestep

        W_dot = self.W_dot()
        self.W += W_dot * timestep

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"V:\n{self.V}, \n W:\n{self.W}\n"
