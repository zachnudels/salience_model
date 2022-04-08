import numpy as np
from typing import Tuple

# np.set_printoptions(threshold=np.inf)


def generate_block(bg_dim: int = 121,
                   figure_dim: int = 25,
                   # TODO: Midpoint change # midpoint: Tuple[int] = (0, 0),
                   figure_orientation: int = 1,
                   bg_orientation: int = 0) -> np.ndarray:
    image = []
    for x in range(bg_dim):
        row = []
        for y in range(bg_dim):
            if (bg_dim / 2 - figure_dim / 2) <= x < bg_dim - (bg_dim / 2 - figure_dim / 2) \
                    and (bg_dim / 2 - figure_dim / 2) <= y < bg_dim - (bg_dim / 2 - figure_dim / 2):
                row.append(figure_orientation)
            else:
                row.append(bg_orientation)
        print(row)
        image.append(row)
    return np.array(image)


if __name__ == '__main__':
    na = generate_block(3, 1)
