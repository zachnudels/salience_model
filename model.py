import numpy as np
from input_processor import InputProcessor
from typing import List


class Model:
    """
    Model expressing the dynamics of figure ground segmentation in the brain as defined in Poort et al. 2012
    The stages of the model are implemented as 3D lists where (0,1,2) corresponds to
    feature preference, x dimension, y dimension  respectively.
    """
    def __init__(self, features: List[int], input_dim: int = 121):
        self.features = features
        self.input_dim = input_dim

        # Construct the model nodes with one activity map for each feature (preference)
        self.preprocessing_stage = [InputProcessor(feature_pref=feature, input_dim=input_dim) for feature in features]

        print(self.preprocessing_stage)

    def update(self, _input: np.ndarray, timestep: float):
        for f in range(len(self.features)):
            self.preprocessing_stage[f].update(_input, timestep)

    def __repr__(self):
        return self.preprocessing_stage.__repr__()

    def __str__(self):
        rtn_string = f"Preprocessing Stage: \n"
        for f in range(len(self.features)):
            rtn_string += f"Feature {self.features[f]}: \n{self.preprocessing_stage[f]}\n"
        return rtn_string






