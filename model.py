import math
import numpy as np
import pandas as pd

from typing import List, Tuple
from functools import partial
from scipy import interpolate, ndimage

from lgn import LGN
from region import Region
from utils import receptive_field_activity


class Model:
    """
    Model expressing the dynamics of figure ground segmentation in the brain as defined in Poort et al. 2012
    The stages of the model are implemented as 3D lists where (0,1,2) corresponds to
    feature preference, x dimension, y dimension  respectively.
    """

    def __init__(self,
                 parameters: pd.DataFrame,
                 features: List[int],
                 input_dim: int = 121,
                 lSquare: int = 25):
        # PARAMETERS

        ratioV1toV2 = 0.5
        ratioV2toV4 = 0.25

        self.time = 0.0
        self.parameters = parameters

        self.features = features
        self.input_dim = input_dim

        V1_dim = input_dim
        V2_dim = round(V1_dim * ratioV1toV2)
        V4_dim = round(V2_dim * ratioV2toV4)

        self.LGN_trace = []
        self.V1_trace = []
        self.V2_trace = []
        self.V4_trace = []

        self.LGN_slice = []
        self.V1_slice = []
        self.V2_slice = []
        self.V4_slice = []

        # Construct the model nodes with one activity map for each feature (preference)
        self.LGN = [LGN(feature_pref=feature, input_dim=input_dim) for feature in features]
        self.V1 = [Region(parameters["V1"], V1_dim, lSquare, V1_dim) for _ in features]
        self.V2 = [Region(parameters["V2"], V2_dim, lSquare, V1_dim) for _ in features]
        self.V4 = [Region(parameters["V4"], V4_dim, lSquare, V1_dim) for _ in features]

        # Instantiate spaces to reduce signal fidelity when feeding forward
        # This is required since a higher region is more coarsely grained, and so we sample the lower region
        # more coarsely corresponding to the ratio between the dimensions of the two regions

        self.V1_Y, self.V1_X = np.meshgrid(np.linspace(-1, 1, V1_dim), np.linspace(-1, 1, V1_dim))
        self.V2_Y, self.V2_X = np.meshgrid(np.linspace(-1, 1, V2_dim), np.linspace(-1, 1, V2_dim))
        self.V4_Y, self.V4_X = np.meshgrid(np.linspace(-1, 1, V4_dim), np.linspace(-1, 1, V4_dim))

        print(f"Initialized model with {len(self.features)} features."
              f"\nV1 has dims {self.V1[0].V.shape}"
              f"\nV2 has dims {self.V2[0].V.shape}"
              f"\nV4 has dims {self.V4[0].V.shape}")

    def update(self, _input: np.ndarray, timestep: float):
        self.time += timestep

        for f in range(len(self.features)):

            self.LGN[f].update(_input, timestep)

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

            V1_2_feedback = feedback_signal(self.V2[f].V,
                                            self.V1[f].fb_kernel,
                                            self.V1_X,
                                            self.V1_Y,
                                            self.V2_X,
                                            self.V2_Y)
            V1_4_feedback = feedback_signal(self.V4[f].V,
                                            self.V1[f].fb_kernel,
                                            self.V1_X,
                                            self.V1_Y,
                                            self.V4_X,
                                            self.V4_Y)
            V1_feedback = V1_2_feedback + V1_4_feedback
            self.V1[f].update_D(V1_feedback, timestep)

            V2_feedback = feedback_signal(self.V4[f].V,
                                          self.V2[f].fb_kernel,
                                          self.V2_X,
                                          self.V2_Y,
                                          self.V4_X,
                                          self.V4_Y)
            self.V2[f].update_D(V2_feedback, timestep)

        # self.record_activity(self.V1, self.V1_trace, self.V1_slice)
        # self.record_activity(self.V2, self.V2_trace, self.V2_slice)
        # self.record_activity(self.V4, self.V4_trace, self.V4_slice)

    def record_activity(self, regions: List[Region], activity_trace: List, activity_slice: List) -> Tuple[List, List]:
        activity_trace.append([
            np.mean(np.array([regions[f].V[regions[f].bck_y, regions[f].bck_x]
                              for f in range(len(self.features))]), axis=0),
            np.mean(np.array([regions[f].V[regions[f].cen_y, regions[f].cen_x]
                              for f in range(len(self.features))]), axis=0),
            np.mean(np.array([regions[f].V[regions[f].brd_y, regions[f].brd_x]
                              for f in range(len(self.features))]), axis=0),
        ])

        activity_slice.append(np.mean(np.array([regions[f].V[regions[f].sliceRow, :]
                              for f in range(len(self.features))]), axis=0))

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
    return interpolate.griddata((lower_X.ravel(), lower_Y.ravel()),
                                activity.ravel(),
                                (higher_X.ravel(), higher_Y.ravel())) \
        .reshape(higher_X.shape)
    # return
    # vectorized_rf_activity = np.vectorize(
    #     partial(receptive_field_activity, activity_map=lower_activity_map, kernel=kernel))
    #
    # return vectorized_rf_activity(X, Y)


def feedback_signal(higher_activity_map: np.ndarray,
                    kernel: np.ndarray,
                    lower_X: np.ndarray,
                    lower_Y: np.ndarray,
                    higher_X: np.ndarray,
                    higher_Y: np.ndarray) -> np.ndarray:
    """
    For the feedback signal, we oversample the higher region's space since it is more coarsely grained simply by
    repeating the activity across both axes in the ratio of the two regions' dimensions
    """
    sampled_data = interpolate.griddata((higher_X.ravel(), higher_Y.ravel()),
                                        higher_activity_map.ravel(),
                                        (lower_X.ravel(), lower_Y.ravel())) \
        .reshape(lower_Y.shape)
    return ndimage.correlate(sampled_data, kernel, mode='nearest')
    # vectorized_rf_activity = np.vectorize(
    #     partial(receptive_field_activity, activity_map=higher_activity_map, kernel=kernel))
    # rfs = vectorized_rf_activity(X, Y)
    #
    # repeats = math.ceil(input_dim_l / input_dim_h)
    # rfs = np.repeat(rfs, repeats, axis=0)
    # rfs = np.repeat(rfs, repeats, axis=1)

    # return rfs
