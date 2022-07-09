import numpy as np
import pandas as pd
import math

from scipy import ndimage
from typing import Dict, Tuple
from utils import gaussian_kernel, runge_kutta2_step, gaussian_1d


class Region:
    """
    Class Representing region unit (with some feature preference) i.e., V1, V2, V4 of the model
    Each unit has four cells (represented as floats) interacting with one another as well as other units
    in higher and lower regions
    """

    def __init__(self,
                 parameters: pd.Series,
                 input_dim: Tuple[int, int],
                 lSquare: int,
                 max_dim: Tuple[int, int],
                 feature_pref: float):
        # PARAMETERS
        self.a = parameters["a"]
        self.c1 = parameters["c1"]
        self.c2 = parameters["c2"]
        self.c3 = parameters["c3"]
        self.c4 = parameters["c4"]
        self.c5 = parameters["c5"]
        self.e1 = parameters["e1"]
        self.e2 = parameters["e2"]
        self.e3 = parameters["e3"]
        self.e4 = parameters["e4"]
        self.e5 = parameters["e5"]
        self.e6 = parameters["e6"]
        self.k1 = parameters["k1"]
        self.k2 = parameters["k2"]
        self.g1 = parameters["g1"]
        self.g2 = parameters["g2"]
        self.g3 = parameters["g3"]
        self.g4 = parameters["g4"]
        self.g5 = parameters["g5"]
        self.g6 = parameters["g6"]
        self.g7 = parameters["g7"]
        self.g8 = parameters["g8"]
        self.g9 = parameters["g9"]

        self.ff_kernel = gaussian_kernel(int(parameters["ff_support"]), parameters["sigma_ff"])
        self.fb_kernel = gaussian_kernel(int(parameters["fb_support"]), parameters["sigma_fb"])

        self.plus_kernel = gaussian_kernel(int(parameters["plus_supp"]), parameters["sigma_plus"])
        self.minus_kernel = gaussian_kernel(int(parameters["minus_supp"]), parameters["sigma_minus"])

        self.similarity_sigma = parameters["similarity_sigma"]
        self.feature_pref = feature_pref

        self.V = np.zeros(shape=input_dim, dtype=np.double)  # Input cell

        self.D = np.zeros(shape=input_dim, dtype=np.double)  # Feedback cell (region filling)

        self.U = np.zeros(shape=input_dim, dtype=np.double)  # Inhibitory cell

        self.W = np.zeros(shape=input_dim, dtype=np.double)  # Center-surround interactions (boundary detection)

        self.S = np.zeros(shape=input_dim, dtype=np.double)  # Similarity interactions (lateral inhibition)

        space_x = np.arange(0, input_dim[0])
        space_y = np.arange(0, input_dim[1])
        self.X, self.Y = np.meshgrid(space_x, space_y)
        self.input_dim = input_dim

        self.bck_x = int(input_dim[0] / 6) - 1
        self.bck_y = int(input_dim[1] / 6) - 1
        self.cen_x = math.ceil(input_dim[0] / 2) - 1
        self.cen_y = math.ceil(input_dim[1] / 2) - 1
        self.brd_x = math.ceil(input_dim[0] / 2) - math.ceil(lSquare / 2 / (max_dim[0] / input_dim[0]))
        self.brd_y = math.ceil(input_dim[1] / 2) - math.ceil(lSquare / 2 / (max_dim[1] / input_dim[1]))
        self.sliceRow = math.ceil(input_dim[0] / 2) - 1

    def V_dot(self, V: np.ndarray, feedforward_signal: np.ndarray) -> np.ndarray:
        """

        :param V:
        :param feedforward_signal:
        :return:
        """
        leak_conductance = (- self.g1 * self.U - self.k2 * self.S) * V  # - 45*opt.U.*(V + 0) ...
        # leak_conductance +=  * V
        driving_input = -self.a * feedforward_signal  # - 5*opt.Gin.

        # There is no effect of D cell in V4 so do not compute for that case
        modulation = 1
        if not pd.isna(self.k1):
            modulation += self.k1 * self.D
            # modulated_driving_input = driving_input * (1 + self.k * self.D) * (V - self.e1)  # *(1 + 2*D).*(V - 5) ...
        # else:

        modulated_driving_input = driving_input * modulation * (V - self.e1)  # *(V - 5)
        boundary_detection = -self.g2 * self.W * (V - self.e2)  # - 1*opt.W.*(V - 2)
        # modulated_driving_input = -self.g8 * self.S *

        V_dot = leak_conductance + modulated_driving_input + boundary_detection
        # divisor = 1 / self.c1
        return V_dot  # * divisor

    def W_dot(self, W: np.ndarray, _input) -> np.ndarray:
        """
        Border detection component
        V - filtered(V) is subtracting the activity of a cell's neighbourhood from its activity
        (in the same feature space). So if a pixel has an orthogonal feature, there will be no activity at that point
        in this activity map and therefore inhibition will be lower. The second filter widens the border so that it gets
        passed up to following layers

        :return:
        """
        leak_conductance = 0 - self.g8 * self.S
        if not pd.isna(self.k1):
            leak_conductance += -self.g3 * self.D  # - 2*opt.Ifb.*V
        else:
            leak_conductance += -self.g3  # - 1/5*V
        leak_conductance *= W
        excitation = -self.g4 * ndimage.correlate(
            abs(_input - ndimage.correlate(_input, self.minus_kernel, mode='nearest')),
            self.plus_kernel, mode='nearest') * (W - self.e3)
        return leak_conductance + excitation

    def U_dot(self, U: np.ndarray, _input=None) -> np.ndarray:
        """
        :return:
        """
        leak_conductance = - (1 / self.c3) * U
        excitatory = - (1 / self.c3) * self.V * (U - self.e4)

        U_dot = leak_conductance + excitatory
        return U_dot

    def D_dot(self, D: np.ndarray, feedback_signal: np.ndarray) -> np.ndarray:
        """
        :param D:
        :param feedback_signal:
        :return:
        """
        leak_conductance = - (1 / self.c4) * D
        feedback_effect = - (1 / self.c4) * feedback_signal * (D - self.e5)

        return leak_conductance + feedback_effect

    def S_dot(self, S: np.ndarray, neighbouring_signals: Dict[float, np.ndarray]) -> np.ndarray:
        leak_conductance = - (1 / self.c5) * S
        similarity = self.g9 * np.sum([gaussian_1d(self.feature_pref - feature_i, 0.0, 27) *
                                       self.V * neighbouring_signals[feature_i]
                                       for feature_i in neighbouring_signals.keys()
                                       if np.sum(self.V * neighbouring_signals[feature_i]) > 0],
                                      axis=0)
        excitatory = - (1 / self.c5) * similarity * (S - self.e6)
        return leak_conductance + excitatory

    def update(self,
               feedforward_signal: np.ndarray,
               timestep: float) -> None:
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
        self.V = runge_kutta2_step(self.V_dot, feedforward_signal, timestep, self.V)

        self.W = runge_kutta2_step(self.W_dot, self.V, timestep, self.W)

        self.U = runge_kutta2_step(self.U_dot, None, timestep, self.U)

    def update_D(self, feedback_signal: np.ndarray, timestep: float) -> None:
        if not pd.isna(self.g7):  # V4 cell - for efficiency don't compute since will be 0
            self.D = runge_kutta2_step(self.D_dot, feedback_signal, timestep, self.D)

    def update_S(self, neighbouring_signal: Dict[float, np.ndarray], timestep: float) -> None:
        self.S = runge_kutta2_step(self.S_dot, neighbouring_signal, timestep, self.S)
