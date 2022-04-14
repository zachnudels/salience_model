import numpy as np
import pandas as pd

from scipy import ndimage

from utils import extract_window


class Region:
    """
    Class Representing region unit (with some feature preference) i.e., V1, V2, V4 of the model
    Each unit has four cells (represented as floats) interacting with one another as well as other units
    in higher and lower regions
    """

    def __init__(self,
                 parameters: pd.Series,
                 input_dim: int):
        # PARAMETERS
        self.a = parameters["a"]
        self.c1 = parameters["c1"]
        self.c2 = parameters["c2"]
        self.c3 = parameters["c3"]
        self.c4 = parameters["c4"]
        self.e1 = parameters["e1"]
        self.e2 = parameters["e2"]
        self.e3 = parameters["e3"]
        self.e4 = parameters["e4"]
        self.e5 = parameters["e5"]
        self.k = parameters["k"]
        self.g1 = parameters["g1"]
        self.g2 = parameters["g2"]
        self.g3 = parameters["g3"]
        self.g4 = parameters["g4"]
        self.g5 = parameters["g5"]
        self.g6 = parameters["g6"]
        self.g7 = parameters["g7"]
        self.sigma_minus = parameters["sigma_minus"]
        self.sigma_plus = parameters["sigma_plus"]
        self.minus_supp = int(parameters["minus_supp"])
        self.plus_supp = int(parameters["plus_supp"])

        self.V = np.zeros(shape=(input_dim, input_dim))  # Input cell
        self.D = np.zeros(shape=(input_dim, input_dim))  # Feedback cell (region filling)
        self.U = np.zeros(shape=(input_dim, input_dim))  # Inhibitory cell
        self.W = np.zeros(shape=(input_dim, input_dim))  # Center-surround interactions (boundary detection)

        # Vectorized version of complex function for boundary detection
        self.v_boundary_activity = np.vectorize(self.boundary_detection_activity)

        space = np.arange(0, input_dim)
        self.Y, self.X = np.meshgrid(space, space)
        self.input_dim = input_dim

    def V_dot(self, feedforward_signal: np.ndarray) -> np.ndarray:
        leak_conductance = -self.g1 * self.U * self.V
        driving_input = self.a * feedforward_signal * (self.e1 - self.V)
        # There is no effect of D cell in V4 so do not compute for that case
        if not pd.isna(self.k):
            modulated_driving_input = driving_input * (1 + self.k * self.D)
        else:
            modulated_driving_input = driving_input
        boundary_detection = self.g2 * self.W * (self.e2 - self.V)

        V_dot = leak_conductance + modulated_driving_input + boundary_detection
        return V_dot / self.c1

    def W_dot(self) -> np.ndarray:
        leak_conductance = -self.g3 * self.D * self.W

        boundary = np.transpose(self.v_boundary_activity(self.X, self.Y))
        excitation = self.g4 * boundary * (self.e3 - self.W)

        W_dot = leak_conductance + excitation
        return W_dot / self.c2

    def U_dot(self) -> np.ndarray:
        leak_conductance = -self.g5 * self.U
        excitatory = self.g6 * self.V * (self.e4 - self.U)

        U_dot = leak_conductance + excitatory
        return U_dot / self.c3

    def D_dot(self, feedback_signal: np.ndarray) -> np.ndarray:
        leak_conductance = -self.g7 * self.D
        feedback_effect = feedback_signal * (self.e5 - self.D)

        D_dot = leak_conductance + feedback_effect
        return D_dot / self.c4

    def update(self, feedforward_signal: np.ndarray, feedback_signal: np.ndarray, timestep: float) -> None:
        V_dot = self.V_dot(feedforward_signal)
        W_dot = self.W_dot()
        U_dot = self.U_dot()
        if not pd.isna(self.g7):  # V4 cell
            D_dot = self.D_dot(feedback_signal)
        else:
            D_dot = None

        self.V += V_dot * timestep
        self.W += W_dot * timestep
        self.U += U_dot * timestep
        if not pd.isna(self.g7):  # V4 cell
            self.D += D_dot * timestep

    def boundary_detection_activity(self, x: int, y: int) -> float:
        window = extract_window(self.V, x, y, self.minus_supp)
        smooth = ndimage.gaussian_filter(window, self.sigma_minus)
        raw_activity = np.abs(window - smooth)

        # filter using the smaller kernel centered on the larger kernel's center
        new_window = extract_window(raw_activity, int(self.minus_supp // 2), int(self.minus_supp // 2), self.plus_supp)
        merged_activity = ndimage.gaussian_filter(new_window, self.sigma_plus)
        return merged_activity[self.plus_supp // 2, self.plus_supp // 2]
