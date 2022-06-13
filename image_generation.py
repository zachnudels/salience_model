import numpy as np
from typing import Tuple, Optional


def generate_block(input_dim: int = 121,
                   figure_dim: Tuple[int] = (25, 25),
                   midpoint: Optional[Tuple[int]] = None,
                   figure_orientation: float = 1,
                   bg_orientation: float = 0) -> np.ndarray:
    image = []
    if midpoint is None:
        midpoint = (input_dim / 2, input_dim / 2)
    for x in range(input_dim):
        row = []
        for y in range(input_dim):
            if (midpoint[0] - figure_dim[0] / 2) <= x <= (midpoint[0] + figure_dim[0] / 2) \
                    and (midpoint[1] - figure_dim[1] / 2) <= y <= (midpoint[1] + figure_dim[1] / 2):
                row.append(figure_orientation)
            else:
                row.append(bg_orientation)
        image.append(row)
    return np.array(image, dtype='float32')

