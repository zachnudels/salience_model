import numpy as np
import pandas as pd
import math

from scipy import ndimage

from utils import gaussian_kernel, runge_kutta2_step


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
                 max_dim: int,
                 similarity_factor: float):
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

        self.plus_kernel = similarity_factor * gaussian_kernel(int(parameters["plus_supp"]), parameters["sigma_plus"])
        self.minus_kernel = similarity_factor * gaussian_kernel(int(parameters["minus_supp"]), parameters["sigma_minus"])

        self.V = np.zeros(shape=(input_dim, input_dim), dtype=np.double)  # Input cell
        self.D = np.zeros(shape=(input_dim, input_dim), dtype=np.double)  # Feedback cell (region filling)
        self.U = np.zeros(shape=(input_dim, input_dim), dtype=np.double)  # Inhibitory cell
        self.W = np.zeros(shape=(input_dim, input_dim), dtype=np.double)  # Center-surround interactions (boundary detection)

        space = np.arange(0, input_dim)
        self.Y, self.X = np.meshgrid(space, space)
        self.input_dim = input_dim

        self.bck_x = int(input_dim / 6) - 1
        self.bck_y = int(input_dim / 6) - 1
        self.cen_x = math.ceil(input_dim/2) - 1
        self.cen_y = math.ceil(input_dim/2) - 1
        self.brd_x = math.ceil(input_dim/2) - math.ceil(lSquare/2 / (max_dim/input_dim))
        self.brd_y = math.ceil(input_dim/2) - math.ceil(lSquare/2 / (max_dim/input_dim))
        self.sliceRow = math.ceil(input_dim / 2) - 1

    def V_dot(self, V: np.ndarray, feedforward_signal: np.ndarray) -> np.ndarray:
        """
        areaV1FF.Gin = areaLGNExc.V;
        areaV1FF.U = areaV1Inh.V;
        areaV1FF.W = areaV1Bd.V;
        areaV1FF.Ifb = areaV1FB.V;

        areaV1FF.f = @(V,opt) ...
        - 45*opt.U.*(V + 0) ...
        - 5*opt.Gin.*(1 + 2*opt.Ifb).*(V - 5) ...
        - 1*opt.W.*(V - 1);

        areaV2FF.f = @(V,opt) ...
        - 45*opt.U.*(V + 0) ...
        - 5*opt.Gin.*(1 + 2*opt.Ifb).*(V - 3) ...
        - 1*opt.W.*(V - 5);

        areaV4FF.f = @(V,opt) ...
        - 25*opt.U.*(V + 0) ...
        - 5*opt.Gin.*(V - 5) ...
        - 1*opt.W.*(V - 2);
        :param V:
        :param feedforward_signal:
        :return:
        """
        leak_conductance = -self.g1 * self.U * V  # - 45*opt.U.*(V + 0) ...
        driving_input = -self.a * feedforward_signal  # - 5*opt.Gin.

        # There is no effect of D cell in V4 so do not compute for that case
        if not pd.isna(self.k):
            modulated_driving_input = driving_input * (1 + self.k * self.D) * (V - self.e1)  # *(1 + 2*D).*(V - 5) ...
        else:
            modulated_driving_input = driving_input * (V - self.e1)  # *(V - 5)
        boundary_detection = -self.g2 * self.W * (V - self.e2)  # - 1*opt.W.*(V - 2)

        V_dot = leak_conductance + modulated_driving_input + boundary_detection
        # divisor = 1 / self.c1
        return V_dot  # * divisor

    def W_dot(self, W: np.ndarray, _input=None) -> np.ndarray:
        """
        Border detection component
        V - filtered(V) is subtracting the activity of a cell's neighbourhood from its activity
        (in the same feature space). So if a pixel has an orthogonal feature, there will be no activity at that point
        in this activity map and therefore inhibition will be lower. The second filter widens the border so that it gets
        passed up to following layers

        areaV1Bd.f = @(V,opt) ...
        - 2*opt.Ifb.*V ...
        - 1/15*imfilter(abs(opt.Gin-imfilter(opt.Gin, LambdaSur, 'same', 'replicate')), ...
        LambdaExc, 'same', 'replicate').*(V - 35);

        areaV2Bd.f = @(V,opt) ...
        - 2*opt.Ifb.*V ...
        - 1/15*imfilter(abs(opt.Gin-imfilter(opt.Gin, LambdaSur, 'same', 'replicate')), ...
        LambdaExc, 'same', 'replicate').*(V - 35);

        areaV4Bd.f = @(V,opt) ...
        - 1/5*V ...
        - 1/5*imfilter(abs(opt.Gin-imfilter(opt.Gin, LambdaSur, 'same', 'replicate')), ...
        LambdaExc, 'same', 'replicate').*(V - A); A = 15 for figure detection

        areaV1Bd.Gin = areaV1FF.V;
        :return:
        """
        if not pd.isna(self.k):
            leak_conductance = -self.g3 * self.D * W  # - 2*opt.Ifb.*V
        else:
            leak_conductance = -self.g3 * W   # - 1/5*V
        excitation = -self.g4 * ndimage.correlate(
            abs(self.V - ndimage.correlate(self.V, self.minus_kernel, mode='nearest')),
            self.plus_kernel, mode='nearest') * (W - self.e3)
        return leak_conductance + excitation

    def U_dot(self, U: np.ndarray, _input=None) -> np.ndarray:
        """
        areaV1Inh.f = @(V,opt) ...
        - 1/5*V ...
        - 1/5*opt.Gin.*(V - 2);

        areaV2Inh.f = @(V,opt) ...
        - 1/5*V ...
        - 1/5*opt.Gin.*(V - 2);

        areaV4Inh.f = @(V,opt) ...
        - 1/5*V ...
        - 1/5*opt.Gin.*(V - 2);

        :return:
        """
        leak_conductance = - (1/self.c3) * U
        excitatory = - (1/self.c3) * self.V * (U - self.e4)

        U_dot = leak_conductance + excitatory
        return U_dot

    def D_dot(self, D: np.ndarray, feedback_signal: np.ndarray) -> np.ndarray:
        """
        areaV1FB.f = @(V,opt) ...
        - 1/15*V ...
        - 1/15*opt.Gin.*(V - B);

        areaV2FB.f = @(V,opt) ...
        - 1/15*V ...
        - 1/15*opt.Gin.*(V - B);   2 for figure detection

        :param D:
        :param feedback_signal:
        :return:
        """
        leak_conductance = - (1/self.c4) * D
        feedback_effect = - (1/self.c4) * feedback_signal * (D - self.e5)

        return leak_conductance + feedback_effect

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
        self.V = runge_kutta2_step(self.V_dot, feedforward_signal, timestep, self.V)

        self.W = runge_kutta2_step(self.W_dot, None, timestep, self.W)

        self.U = runge_kutta2_step(self.U_dot, None, timestep, self.U)

    def update_D(self, feedback_signal: np.ndarray, timestep: float) -> None:
        if not pd.isna(self.g7):  # V4 cell - for efficiency don't compute since will be 0
            self.D = runge_kutta2_step(self.D_dot, feedback_signal, timestep, self.D)

