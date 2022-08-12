import pandas as pd
import numpy as np
from image_generation import generate_2_blocks

from model import Model


def run(orientations, parm_df):
    stimuli = {"vheus_exp_2": {"V4": {"bg": (8, 10, 10, 12), "cen1": (2, 4, 4, 6), "cen2": (15, 17, 17, 19)},
                               "V1": {"bg": (80, 96, 64, 80), "cen1": (15, 29, 33, 45), "cen2": (121, 135, 138, 150)}},
               "vheus_exp_1": {"FEF": {"bg": (1, 3, 1, 3), "cen1": (8, 10, 8, 10), "cen2": (13, 15, 13, 15)},
                               "V4": {"bg": (1, 3, 1, 3), "cen1": (8, 10, 8, 10), "cen2": (13, 15, 13, 15)},
                               "V2": {"bg": (5, 10, 5, 10), "cen1": (32, 38, 32, 38), "cen2": (53, 59, 53, 59)},
                               "V1": {"bg": (10, 20, 10, 20), "cen1": (64, 76, 64, 76), "cen2": (105, 117, 105, 117)}}}

    bg = orientations[0]
    fg1 = orientations[1]
    fg2 = orientations[2]
    V1_dim = (183, 183)

    na = generate_2_blocks(input_dim=V1_dim,
                           bg_orientation=bg,
                           figure_orientations=[fg1, fg2],
                           figure_dim=(12, 12),
                           midpoints=[(70, 70), (111, 111)])

    model = Model(parm_df,
                  input_dim=V1_dim,
                  features=[bg, fg1, fg2])
    n = 600
    empty = np.full_like(na, -1)

    cen1 = []
    cen2 = []
    for i in range(n):
        if i < 40:
            model.update(empty, 10e-3)
        else:
            model.update(na, 10e-3)

        FEF = np.mean(np.array([model.FEF[f].V for f in range(len(model.features))]), axis=0)

        layer = "FEF"
        stim = stimuli["vheus_exp_1"][layer]
        cen1_i = stim["cen1"]
        cen2_i = stim["cen2"]

        cen1.append(np.mean(FEF[cen1_i[0]:cen1_i[1], cen1_i[2]:cen1_i[3]]))
        cen2.append(np.mean(FEF[cen2_i[0]:cen2_i[1], cen2_i[2]:cen2_i[3]]))

    return [cen1, cen2]
