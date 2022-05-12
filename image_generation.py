import numpy as np


def generate_block(input_dim: int = 121,
                   figure_dim: int = 25,
                   # TODO: Midpoint change # midpoint: Tuple[int] = (0, 0),
                   figure_orientation: float = 1,
                   bg_orientation: float = 0) -> np.ndarray:
    image = []
    for x in range(input_dim):
        row = []
        for y in range(input_dim):
            if (input_dim / 2 - figure_dim / 2) <= x < input_dim - (input_dim / 2 - figure_dim / 2) \
                    and (input_dim / 2 - figure_dim / 2) <= y < input_dim - (input_dim / 2 - figure_dim / 2):
                row.append(figure_orientation)
            else:
                row.append(bg_orientation)
        image.append(row)
    return np.array(image, dtype='float32')


if __name__ == '__main__':
    na = generate_block(120)
