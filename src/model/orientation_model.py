import numpy as np
import pandas as pd

from functools import partial
from scipy.stats import vonmises
from typing import Callable, Dict, List, Tuple, Union

from .model import Model


class OrientationModel(Model):
    def __init__(
        self,
        parameters: pd.DataFrame,
        channels: int,
        input_dim: Tuple[int, int],
        input_activity: np.ndarray = None,
        recording_sites: Dict[str, Dict[str, Tuple[int, int]]] = None,
        initial_recordings: Dict[str, List[float]] = None,
    ):
        """
        A salience_model simulating cortical activity in the human visual system

        :param parameters: parameters information used by the salience_model
        :param channels: number of tuning curves or channels
        :param input_dim: input dimension of the salience_model
        :param input_activity: input image (if this is None, it must be set in the salience_model.update() method).
        :param recording_sites: the names and locations (X,Y) of where we want to record activity from for each layer
        :param initial_recordings: the recording_sites names and initial lists of floats to add on to the beginning of
                                   our recording arrays
        """
        features = np.arange(0, 180, 180 / channels)
        super().__init__(
            parameters,
            features,
            tuning_function(len(features)),
            input_dim,
            input_activity,
            recording_sites,
            initial_recordings,
        )


def tuning_function(
    channels: int,
) -> Callable[[Union[List, float], float], Union[List, float]]:
    """
    Given a number of channels, this function returns a tuning function used by the orientation salience_model.
    It finds parameters to use with the Von Mises distribution. We desire that the normalized sum of each tuning curve
    along the orientation space sum as close to 1 as possible within some margin (which we set to 1e-6). This will
    reduce the disturbances in the salience_model since these tiny perturbations will amplify as the salience_model updates.
    :param channels: number of orientation channels used by the salience_model
    :return: A tuning function used by the salience_model
    """
    kappa = channels
    max_diff = 10
    x = np.arange(0, 180)
    norm = 1

    while max_diff > 1e-6:
        if kappa == 1:
            break
        y = []
        for j in np.arange(0, 180, 180 / channels):
            channel = tuning_curve(x, j, kappa)
            y.append(channel)

        y = np.sum(y, axis=0)
        y_max = np.max(y)

        y = []
        for j in np.arange(0, 180, 180 / channels):
            channel = tuning_curve(x, j, kappa) / y_max
            y.append(channel)

        y = np.sum(y, axis=0)
        max_diff = np.max(np.ediff1d(y))

        norm = y_max
        kappa -= 0.1

    return partial(tuning_curve, kappa=kappa, norm=norm)


def to_rad(degrees):
    return (degrees / 180) * np.pi


def tuning_curve(signal, preference, kappa, norm=1):
    """
    The tuning function used in our experiment is a Von Mises distribution. This distribution has been shown to
    acceptably approximate orientation tuning curves in cat [Swindale, 1997]
    :param signal:
    :param preference:
    :param kappa:
    :param norm:
    :return:
    """
    return vonmises.pdf(to_rad(signal), kappa, to_rad(preference), 0.5) / norm
