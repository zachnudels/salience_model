import pandas as pd

from model import Model
from image_generation import generate_block

if __name__ == '__main__':
    parm_df = pd.read_csv("parameters.csv", index_col=0)
    model = Model(parm_df, features=[0, 1], input_dim=120)
    na = generate_block(input_dim=120)
    some_cell_v = []
    some_cell_w = []
    for i in range(2):
        #     print(model.preprocessing_stage[1][1][1].v)
        # some_cell_v.extend(model.V1[1].V)
        # some_cell_w.extend(model.V1[1].W)
        model.update(na, 10e-3)
