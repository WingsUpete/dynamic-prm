import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

__all__ = ['plot_circle']


def plot_circle(x: float, y: float, r: float, c: str = 'b', fill: bool = True) -> None:
    """
    Draw a large circle of given color.
    :param x: x position of the circle
    :param y: y position of the circle
    :param r: radius of the circle
    :param c: color
    :param fill: fill the circle
    """
    if fill:
        filled_circle = Circle((x, y), radius=r, color=c)
        plt.gca().add_patch(filled_circle)
    else:
        deg_list = list(range(0, 360, 5))
        deg_list.append(0)
        x_list = [x + r * math.cos(np.deg2rad(d)) for d in deg_list]
        y_list = [y + r * math.sin(np.deg2rad(d)) for d in deg_list]
        plt.plot(x_list, y_list, f'-{c}')
