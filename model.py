import math
import numpy as np
import pandas as pd

from typing import List

from input_processor import InputProcessor
from region import Region

from utils import feedforward_signal, feedback_signal


class Model:
    """
    Model expressing the dynamics of figure ground segmentation in the brain as defined in Poort et al. 2012
    The stages of the model are implemented as 3D lists where (0,1,2) corresponds to
    feature preference, x dimension, y dimension  respectively.
    """

    def __init__(self,
                 parameters: pd.DataFrame,
                 features: List[int],
                 input_dim: int = 120):
        # PARAMETERS
        self.parameters = parameters

        self.features = features
        self.input_dim = input_dim

        V1_dim = input_dim
        V2_dim = math.ceil(input_dim / 2)
        V4_dim = math.floor(math.ceil(input_dim / 2) / 4)

        # Construct the model nodes with one activity map for each feature (preference)
        self.preprocessing_stage = [InputProcessor(feature_pref=feature, input_dim=input_dim) for feature in features]
        self.V1 = [Region(parameters["V1"], V1_dim) for _ in features]
        self.V2 = [Region(parameters["V2"], V2_dim) for _ in features]
        self.V4 = [Region(parameters["V4"], V4_dim) for _ in features]

        V1_V2_ff_space = np.arange(0, V1_dim, math.ceil(V1_dim / V2_dim))
        self.V1_V2_Y, self.V1_V2_X = np.meshgrid(V1_V2_ff_space, V1_V2_ff_space)

        V2_V4_ff_space = np.arange(0, V2_dim, math.ceil(V2_dim / V4_dim))
        self.V2_V4_Y, self.V2_V4_X = np.meshgrid(V2_V4_ff_space, V2_V4_ff_space)

        print(f"Initialized model with {len(self.features)} features."
              f"\nV1 has dims {self.V1[0].V.shape}"
              f"\nV2 has dims {self.V2[0].V.shape}"
              f"\nV4 has dims {self.V4[0].V.shape}")

    def update(self, _input: np.ndarray, timestep: float):
        for f in range(len(self.features)):
            V1_feedforward = self.preprocessing_stage[f].V
            V1_2_feedback = feedback_signal(self.V2[f].V,
                                            self.V2[f].input_dim,
                                            self.V1[f].input_dim,
                                            self.parameters["V1"]["fb_support"],
                                            self.parameters["V1"]["sigma_fb"],
                                            self.V2[f].X,
                                            self.V2[f].Y)
            V1_4_feedback = feedback_signal(self.V4[f].V,
                                            self.V4[f].input_dim,
                                            self.V1[f].input_dim,
                                            self.parameters["V1"]["fb_support"],
                                            self.parameters["V1"]["sigma_fb"],
                                            self.V4[f].X,
                                            self.V4[f].Y)
            V1_feedback = V1_2_feedback + V1_4_feedback

            V2_feedforward = feedforward_signal(self.V1[f].V,
                                                self.parameters["V2"]["ff_support"],
                                                self.parameters["V2"]["sigma_ff"],
                                                self.V1_V2_X,
                                                self.V1_V2_Y)
            V2_feedback = feedback_signal(self.V4[f].V,
                                          self.V4[f].input_dim,
                                          self.V2[f].input_dim,
                                          self.parameters["V2"]["fb_support"],
                                          self.parameters["V2"]["sigma_fb"],
                                          self.V4[f].X,
                                          self.V4[f].Y)

            V4_feedforward = feedforward_signal(self.V2[f].V,
                                                self.parameters["V4"]["ff_support"],
                                                self.parameters["V4"]["sigma_ff"],
                                                self.V2_V4_X,
                                                self.V2_V4_Y)

            self.preprocessing_stage[f].update(_input, timestep)
            self.V1[f].update(V1_feedforward, V1_feedback, timestep)
            self.V2[f].update(V2_feedforward, V2_feedback, timestep)
            self.V4[f].update(V4_feedforward, np.zeros_like(V4_feedforward), timestep)

    def __repr__(self):
        return self.preprocessing_stage.__repr__()

    def __str__(self):
        rtn_string = f"Preprocessing Stage: \n"
        for f in range(len(self.features)):
            rtn_string += f"Feature {self.features[f]}: \n{self.preprocessing_stage[f]}\n"
        return rtn_string
