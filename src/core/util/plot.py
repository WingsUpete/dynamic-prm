import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

__all__ = ['plot_circle', 'draw_path', 'draw_query_points']


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


def draw_path(path: list[list[float]], c: str = 'm') -> None:
    """
    Draws the path as solid line with specified color.
    :param path: specified path
    :param c: specified color (magenta by default)
    """
    plt.plot([x for (x, _) in path], [y for (_, y) in path], f'-{c}')  # - = solid line


def draw_query_points(start: list[float], goal: list[float], c: str = 'r') -> None:
    """
    Draws the starting point and the goal point for a query
    :param start: coordinate list of the starting point
    :param goal: coordinate list of the goal point
    :param c: color of the query points
    """
    plt.plot(start[0], start[1], f'^{c}')  # ^ = triangle
    plt.plot(goal[0], goal[1], f'd{c}')  # d = thin_diamond marker
