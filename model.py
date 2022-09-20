import math

import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Callable, Dict, List, Tuple, Union
from scipy import interpolate, ndimage

from lgn import LGN
from fef import FEF
from layer import Layer


class Model:
    """
    Model expressing the dynamics of figure ground segmentation in the brain as defined in Poort et al. 2012
    The stages of the model are implemented as 3D lists where (0,1,2) corresponds to
    feature preference, x dimension, y dimension  respectively.
    """

    def __init__(self,
                 parameters: pd.DataFrame,
                 features: Union[List[int], np.ndarray],
                 tuning_curve: Callable,
                 input_dim: Tuple[int, int] = (121, 121),
                 input_activity: np.ndarray = None,
                 recording_sites: Dict[str, Dict[str, Tuple[int, int]]] = None,
                 initial_recordings: Dict[str, List[float]] = None,
                 ):

        # PARAMETERS
        self.tuning_curve = tuning_curve
        ratioV1toV2 = 0.5
        ratioV2toV4 = 0.25

        self.parameters = parameters

        self.features = features
        self.input_dim = input_dim[1], input_dim[0]

        V1_dim = self.input_dim
        V2_dim = tuple(math.ceil(V1_dim_i * ratioV1toV2) for V1_dim_i in V1_dim)
        V4_dim = tuple(math.floor(V2_dim_i * ratioV2toV4) for V2_dim_i in V2_dim)

        # Construct the model nodes with one activity map for each feature (preference)
        self.LGN = LGN(features, self.input_dim)
        self.V1 = Layer(parameters["V1"], V1_dim, features, tuning_curve)
        self.V2 = Layer(parameters["V2"], V2_dim, features, tuning_curve)
        self.V4 = Layer(parameters["V4"], V4_dim, features, tuning_curve)

        self.FEF = FEF(parameters["FEF"], features, V4_dim)

        # Instantiate spaces to reduce signal fidelity when feeding forward
        # This is required since a higher region is more coarsely grained, and so we sample the lower region
        # more coarsely corresponding to the ratio between the dimensions of the two regions
        self.V1_X, self.V1_Y = np.linspace(-1, 1, V1_dim[1]), np.linspace(-1, 1, V1_dim[0])
        self.V2_X, self.V2_Y = np.linspace(-1, 1, V2_dim[1]), np.linspace(-1, 1, V2_dim[0])
        self.V4_X, self.V4_Y = np.linspace(-1, 1, V4_dim[1]), np.linspace(-1, 1, V4_dim[0])

        self.input_activity = np.zeros(shape=(V1_dim[0], V1_dim[1], len(self.features)))
        # If the only one input is used in the simulation, process it now to save computation time
        if input_activity is not None:
            self.process_input(input_activity)

        self.recording_sites = recording_sites
        if initial_recordings is None:
            initial_recordings = []

        self.recordings = {layer: {site: deepcopy(initial_recordings) for site in recording_sites[layer]}
                           for layer in recording_sites}

    def process_input(self, activity):
        for k in range(len(self.features)):
            self.input_activity[:, :, k] = self.tuning_curve(activity, self.features[k])

    def update(self, timestep: float, bu_input: np.ndarray = None, td_input: np.ndarray = None):

        if bu_input is not None:
            self.process_input(bu_input)

        self.LGN.update(self.input_activity, timestep)

        V1_feedforward = self.LGN.V
        self.V1.update(V1_feedforward, timestep)

        V2_feedforward = feedforward_signal(self.V1.V,
                                            self.V2.ff_kernel,
                                            self.V1_X,
                                            self.V1_Y,
                                            self.V2_X,
                                            self.V2_Y)
        self.V2.update(V2_feedforward, timestep)

        V4_feedforward = feedforward_signal(self.V2.V,
                                            self.V4.ff_kernel,
                                            self.V2_X,
                                            self.V2_Y,
                                            self.V4_X,
                                            self.V4_Y)
        self.V4.update(V4_feedforward, timestep)

        FEF_feedforward = self.V4.V
        self.FEF.update(FEF_feedforward, timestep)

        V1_2_feedback = feedback_interp(self.V2.W,
                                        self.V1_X,
                                        self.V1_Y,
                                        self.V2_X,
                                        self.V2_Y)
        V1_4_feedback = feedback_interp(self.V4.W,
                                        self.V1_X,
                                        self.V1_Y,
                                        self.V4_X,
                                        self.V4_Y)
        V1_feedback = np.zeros_like(self.V1.V)
        for i in range(len(self.features)):
            V1_feedback[:, :, i] = ndimage.correlate(V1_2_feedback[:, :, i] + V1_4_feedback[:, :, i],
                                                     self.V1.fb_kernel,
                                                     mode='nearest')
        self.V1.update_D(V1_feedback, timestep)

        _V2_feedback = feedback_interp(self.V4.W,
                                       self.V2_X,
                                       self.V2_Y,
                                       self.V4_X,
                                       self.V4_Y)
        V2_feedback = np.zeros_like(self.V2.W)
        for i in range(len(self.features)):
            V2_feedback[:, :, i] = ndimage.correlate(_V2_feedback[:, :, i],
                                                     self.V2.fb_kernel,
                                                     mode='nearest')
        self.V2.update_D(V2_feedback, timestep)

        if td_input is not None:
            self.V4.update(td_input, timestep)

        for layer in self.recording_sites:
            mean_activity = np.mean(getattr(self, layer).V, axis=2)
            for site in self.recording_sites[layer]:
                self.recordings[layer][site].append(mean_activity[self.recording_sites[layer][site]])

    def simulate(self, n=600, timestep=10e-3):
        for i in range(n):
            self.update(timestep)
        return self.recordings


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
    higher_activity = np.zeros((higher_X.shape[0], higher_Y.shape[0], lower_activity_map.shape[2]))
    for i in range(lower_activity_map.shape[2]):
        activity = ndimage.correlate(lower_activity_map[:, :, i], kernel, mode='nearest')
        f = interpolate.interp2d(lower_X, lower_Y, activity)
        higher_activity[:, :, i] = f(higher_X, higher_Y)
    return higher_activity


def feedback_interp(higher_activity_map: np.ndarray,
                    lower_X: np.ndarray,
                    lower_Y: np.ndarray,
                    higher_X: np.ndarray,
                    higher_Y: np.ndarray) -> np.ndarray:
    """
    For the feedback signal, we oversample the higher region's space since it is more coarsely grained simply by
    repeating the activity across both axes in the ratio of the two regions' dimensions

    """
    lower_activity = np.zeros((lower_X.shape[0], lower_Y.shape[0], higher_activity_map.shape[2]))
    for i in range(higher_activity_map.shape[2]):
        f = interpolate.interp2d(higher_X, higher_Y, higher_activity_map[:, :, i])
        lower_activity[:, :, i] = f(lower_X, lower_Y)
    return lower_activity
