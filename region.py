import numpy as np
import pandas as pd
import math
from functools import partial
from scipy import ndimage

from utils import gaussian_kernel, receptive_field_activity


class Region:
    """
    Class Representing region unit (with some feature preference) i.e., V1, V2, V4 of the model
    Each unit has four cells (represented as floats) interacting with one another as well as other units
    in higher and lower regions
    """

    def __init__(self,
                 parameters: pd.Series,
                 input_dim: int,
                 lSquare: int,
                 max_dim: int):
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

        self.ff_kernel = gaussian_kernel(int(parameters["ff_support"]), parameters["sigma_ff"])
        self.fb_kernel = gaussian_kernel(int(parameters["fb_support"]), parameters["sigma_fb"])

        self.plus_kernel = gaussian_kernel(int(parameters["plus_supp"]), parameters["sigma_plus"])
        self.minus_kernel = gaussian_kernel(int(parameters["minus_supp"]), parameters["sigma_minus"])

        self.V = np.zeros(shape=(input_dim, input_dim))  # Input cell
        self.D = np.zeros(shape=(input_dim, input_dim))  # Feedback cell (region filling)
        self.U = np.zeros(shape=(input_dim, input_dim))  # Inhibitory cell
        self.W = np.zeros(shape=(input_dim, input_dim))  # Center-surround interactions (boundary detection)

        space = np.arange(0, input_dim)
        self.Y, self.X = np.meshgrid(space, space)
        self.input_dim = input_dim

        self.bck_x = int(input_dim / 6)
        self.bck_y = int(input_dim / 6)
        self.cen_x = math.ceil(input_dim/2)
        self.cen_y = math.ceil(input_dim/2)
        self.brd_x = math.ceil(input_dim/2) - math.ceil(lSquare/2 / (max_dim/input_dim)) + 1
        self.brd_y = math.ceil(input_dim/2)
        self.sliceRow = math.ceil(input_dim / 2)

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
        divisor = 1 / self.c1
        return V_dot * divisor

    def W_dot(self) -> np.ndarray:
        """
        Border detection component
        V - filtered(V) is subtracting the activity of a cell's neighbourhood from its activity
        (in the same feature space). So if a pixel has an orthogonal feature, there will be no activity at that point
        in this activity map and therefore inhibition will be lower. The second filter widens the border so that it gets
        passed up to following layers

        :return:
        """
        leak_conductance = -self.g3 * self.D * self.W
        excitation = self.g4 * ndimage.correlate(
            abs(self.V - ndimage.correlate(self.V, self.minus_kernel, mode='nearest')),
            self.plus_kernel, mode='nearest') * (self.e3 - self.W)
        W_dot = leak_conductance + excitation
        divisor = 1 / self.c2
        return W_dot * divisor

    def U_dot(self) -> np.ndarray:
        leak_conductance = -self.g5 * self.U
        excitatory = self.g6 * self.V * (self.e4 - self.U)

        U_dot = leak_conductance + excitatory
        divisor = 1 / self.c3
        return U_dot * divisor

    def D_dot(self, feedback_signal: np.ndarray) -> np.ndarray:
        leak_conductance = -self.g7 * self.D
        feedback_effect = feedback_signal * (self.e5 - self.D)

        D_dot = leak_conductance + feedback_effect
        divisor = 1 / self.c4
        return D_dot * divisor

    def record_activity(self):
        self.activity_trace.append([
            np.mean(self.V[self.bck_x, self.bck_y]),
            np.mean(self.V[self.bck_x, self.bck_y])
        ])

    def update(self, feedforward_signal: np.ndarray, timestep: float) -> None:
        """
        areaV1FF.Gin = areaLGNExc.V; feedforward signal
        areaV1FF.U = areaV1Inh.V;
        areaV1FF.W = areaV1Bd.V;
        areaV1FF.Ifb = areaV1FB.V;
        areaV1FF = updateNeuronField(areaV1FF);
        areaV1Bd.Ifb = areaV1FB.V;
        areaV1Bd.Gin = areaV1FF.V;
        areaV1Bd = updateNeuronField(areaV1Bd);
        areaV1Inh.Ifb = areaV1FB.V;
        areaV1Inh.Gin = areaV1FF.V;
        areaV1Inh = updateNeuronField(areaV1Inh);
        """
        V_dot = self.V_dot(feedforward_signal)
        self.V += V_dot * timestep

        W_dot = self.W_dot()
        self.W += W_dot * timestep

        U_dot = self.U_dot()
        self.U += U_dot * timestep

    def update_D(self, feedback_signal: np.ndarray, timestep: float) -> None:
        if not pd.isna(self.g7):  # V4 cell - for efficiency don't compute since will be 0
            D_dot = self.D_dot(feedback_signal)
        else:
            D_dot = None
        if not pd.isna(self.g7):  # V4 cell - for efficiency don't compute since will be 0
            self.D += D_dot * timestep
