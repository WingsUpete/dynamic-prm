import math

__all__ = ['cal_dist', 'cal_dist_n_angle']


def cal_dist(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """
    Calculates the distance and angle from one node to the other.
    :param from_x: x coordinate of the `from_node`
    :param from_y: y coordinate of the `from_node`
    :param to_x: x coordinate of the `to_node`
    :param to_y: y coordinate of the `to_node`
    :return: `d` as distance
    """
    return math.hypot(to_x - from_x, to_y - from_y)


def cal_dist_n_angle(from_x: float, from_y: float,
                     to_x: float, to_y: float) -> (float, float):
    """
    Calculates the distance and angle from one node to the other.
    :param from_x: x coordinate of the `from_node`
    :param from_y: y coordinate of the `from_node`
    :param to_x: x coordinate of the `to_node`
    :param to_y: y coordinate of the `to_node`
    :return: `d` as distance, `theta` as angle
    """
    dx = to_x - from_x
    dy = to_y - from_y
    d = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)
    return d, theta
