import numpy as np
from typing import Tuple, Optional, List


def generate_block(input_dim: Tuple[int, int] = (121, 121),
                   figure_dim: Tuple[int, int] = (25, 25),
                   midpoint: Optional[Tuple[int, int]] = None,
                   figure_orientation: float = 1,
                   bg_orientation: float = 0) -> np.ndarray:
    image = np.full(input_dim, bg_orientation)

    if midpoint is None:
        mid_x, mid_y = int(input_dim[1] / 2), int(input_dim[0] / 2)
    else:
        mid_x = midpoint[1]
        mid_y = midpoint[0]
    fig_x = int(figure_dim[1] / 2)
    fig_y = int(figure_dim[0] / 2)

    if figure_dim[1] % 2 != 0 and figure_dim[0] % 2 != 0:
        image[mid_x - fig_x: mid_x + fig_x + 1, mid_y - fig_y: mid_y + fig_y + 1] = figure_orientation
    else:
        image[mid_x - fig_x: mid_x + fig_x, mid_y - fig_y: mid_y + fig_y] = figure_orientation

    return image


def generate_2_blocks(input_dim: Tuple[int, int] = (121, 121),
                      figure_dim: Tuple[int, int] = (25, 25),
                      figure_orientations: List[float] = None,
                      bg_orientation: float = 0,
                      midpoints: List[Tuple[int, int]] = None) -> np.ndarray:
    image = np.full((input_dim[1], input_dim[0]), bg_orientation)
    fig_x = int(figure_dim[1]/2)
    fig_y = int(figure_dim[0]/2)
    mid1_x, mid1_y = midpoints[0][1], midpoints[0][0]
    mid2_x, mid2_y = midpoints[1][1], midpoints[1][0]

    if fig_x == 0 and fig_y != 0:
        image[mid1_x, mid1_y-fig_y: mid1_y+fig_y] = figure_orientations[0]
    elif fig_x != 0 and fig_y == 0:
        image[mid1_x-fig_x: mid1_x+fig_x, mid1_y] = figure_orientations[0]
    elif fig_x == 0 and fig_y == 0:
        image[mid1_x, mid1_y] = figure_orientations[0]
    else:
        image[mid1_x-fig_x: mid1_x+fig_x, mid1_y-fig_y: mid1_y+fig_y] = figure_orientations[0]

    if fig_x == 0 and fig_y != 0:
        image[mid2_x, mid2_y-fig_y: mid2_y+fig_y] = figure_orientations[1]
    elif fig_x != 0 and fig_y == 0:
        image[mid2_x-fig_x: mid2_x+fig_x, mid2_y] = figure_orientations[1]
    elif fig_x == 0 and fig_y == 0:
        image[mid2_x, mid2_y] = figure_orientations[1]
    else:
        image[mid2_x-fig_x: mid2_x+fig_x, mid2_y-fig_y: mid2_y+fig_y] = figure_orientations[1]
    return image
