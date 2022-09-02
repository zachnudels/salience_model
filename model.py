import math
import numpy as np
import pandas as pd

from typing import List, Tuple
from scipy import interpolate, ndimage

from lgn import LGN
from fef import FEF
from region import Region


class Model:
    """
    Model expressing the dynamics of figure ground segmentation in the brain as defined in Poort et al. 2012
    The stages of the model are implemented as 3D lists where (0,1,2) corresponds to
    feature preference, x dimension, y dimension  respectively.
    """

    def __init__(self,
                 parameters: pd.DataFrame,
                 features: List[int],
                 input_dim: Tuple[int, int] = (121, 121),
                 lSquare: int = 25):  # TODO: Change code so can set where cen, bck, etc. lies
        # PARAMETERS

        ratioV1toV2 = 0.5
        ratioV2toV4 = 0.25

        self.parameters = parameters

        self.features = features
        self.input_dim = input_dim[1], input_dim[0]

        V1_dim = self.input_dim
        V2_dim = tuple(math.ceil(V1_dim_i * ratioV1toV2) for V1_dim_i in V1_dim)
        V4_dim = tuple(math.floor(V2_dim_i * ratioV2toV4) for V2_dim_i in V2_dim)

        # Construct the model nodes with one activity map for each feature (preference)
        self.LGN = [LGN(feature, self.input_dim,) for feature in features]
        self.V1 = [Region(parameters["V1"], V1_dim, lSquare, V1_dim, feature) for feature in features]
        self.V2 = [Region(parameters["V2"], V2_dim, lSquare, V1_dim, feature) for feature in features]
        self.V4 = [Region(parameters["V4"], V4_dim, lSquare, V1_dim, feature) for feature in features]
        self.FEF = [FEF(parameters["FEF"], feature, V4_dim) for feature in features]

        # Instantiate spaces to reduce signal fidelity when feeding forward
        # This is required since a higher region is more coarsely grained, and so we sample the lower region
        # more coarsely corresponding to the ratio between the dimensions of the two regions
        self.V1_X, self.V1_Y = np.linspace(-1, 1, V1_dim[1]), np.linspace(-1, 1, V1_dim[0])
        self.V2_X, self.V2_Y = np.linspace(-1, 1, V2_dim[1]), np.linspace(-1, 1, V2_dim[0])
        self.V4_X, self.V4_Y = np.linspace(-1, 1, V4_dim[1]), np.linspace(-1, 1, V4_dim[0])

    def update(self, _input: np.ndarray, timestep: float):
        for f in range(len(self.features)):
            self.LGN[f].update(_input, timestep)

            V1_neighbours = {self.V1[_f].feature_pref: self.V1[_f].V for _f in range(len(self.features)) if _f != f}
            self.V1[f].update_S(V1_neighbours, timestep)

            V2_neighbours = {self.V2[_f].feature_pref: self.V2[_f].V for _f in range(len(self.features)) if _f != f}
            self.V2[f].update_S(V2_neighbours, timestep)

            V4_neighbours = {self.V4[_f].feature_pref: self.V4[_f].V for _f in range(len(self.features)) if _f != f}
            self.V4[f].update_S(V4_neighbours, timestep)

            V1_feedforward = self.LGN[f].V
            self.V1[f].update(V1_feedforward, timestep)

            V2_feedforward = feedforward_signal(self.V1[f].V,
                                                self.V2[f].ff_kernel,
                                                self.V1_X,
                                                self.V1_Y,
                                                self.V2_X,
                                                self.V2_Y)
            self.V2[f].update(V2_feedforward, timestep)

            V4_feedforward = feedforward_signal(self.V2[f].V,
                                                self.V4[f].ff_kernel,
                                                self.V2_X,
                                                self.V2_Y,
                                                self.V4_X,
                                                self.V4_Y)
            self.V4[f].update(V4_feedforward, timestep)

            FEF_feedforward = self.V4[f].V
            self.FEF[f].update(FEF_feedforward, timestep)

            V1_2_feedback = feedback_interp(self.V2[f].W,
                                            self.V1_X,
                                            self.V1_Y,
                                            self.V2_X,
                                            self.V2_Y)
            V1_4_feedback = feedback_interp(self.V4[f].W,
                                            self.V1_X,
                                            self.V1_Y,
                                            self.V4_X,
                                            self.V4_Y)
            V1_feedback = ndimage.correlate(V1_2_feedback + V1_4_feedback, self.V1[f].fb_kernel, mode='nearest')
            self.V1[f].update_D(V1_feedback, timestep)

            V2_feedback = feedback_interp(self.V4[f].W,
                                          self.V2_X,
                                          self.V2_Y,
                                          self.V4_X,
                                          self.V4_Y)
            V2_feedback = ndimage.correlate(V2_feedback, self.V2[f].fb_kernel, mode='nearest')
            self.V2[f].update_D(V2_feedback, timestep)

    def __repr__(self):
        return self.LGN.__repr__()

    def __str__(self):
        rtn_string = f"Preprocessing Stage: \n"
        for f in range(len(self.features)):
            rtn_string += f"Feature {self.features[f]}: \n{self.LGN[f]}\n"
        return rtn_string


def feedforward_signal(lower_activity_map: np.ndarray,
                       kernel: np.ndarray,
                       lower_X: np.ndarray,
                       lower_Y: np.ndarray,
                       higher_X: np.ndarray,
                       higher_Y: np.ndarray) -> np.ndarray:
    """
    For each unit (x,y) in X,Y sampled from the lower region, calculate the RF to generate the signal that will
    be fed forward to the subsequent region

    """
    activity = ndimage.correlate(lower_activity_map, kernel, mode='nearest')
    f = interpolate.interp2d(lower_X, lower_Y, activity)
    return f(higher_X, higher_Y)


def feedback_interp(higher_activity_map: np.ndarray,
                    lower_X: np.ndarray,
                    lower_Y: np.ndarray,
                    higher_X: np.ndarray,
                    higher_Y: np.ndarray) -> np.ndarray:
    """
    For the feedback signal, we oversample the higher region's space since it is more coarsely grained simply by
    repeating the activity across both axes in the ratio of the two regions' dimensions

    """
    f = interpolate.interp2d(higher_X, higher_Y, higher_activity_map)
    return f(lower_X, lower_Y)

