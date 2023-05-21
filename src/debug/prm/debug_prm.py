import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../'))  # load core
import math
import random
import time

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info('Initializing...')

import matplotlib.pyplot as plt

from core.prm import *


def gen_map_edge_obstacles(map_range: list[float], robot_radius: float,
                           shrink_factor: float = 0.9) -> (list[float], list[float], list[float]):
    """
    Generates point obstacles at the edge of the map, ensuring that the robot cannot leave the map.

    Suppose the obstacle interval (distance between two consecutive obstacles) is `d`, then it requires that `d < 2r`.
    Using a shrink factor `0 << t < 1`, we can write `d <= 2tr`. Now suppose the map range is of `l` length, then the
    number of obstacles we have to place is `n = l/d + 1 >= l/2tr + 1`, which leads to `n = ceil(l/2tr) + 1`.
    :param map_range: the range of the map, as `[min, max]` for both `x` and `y` (`max - min = l`)
    :param robot_radius: the radius `r` of the round robot
    :param shrink_factor: how much should the obstacle interval instance be shrunk (`t`) to avoid the robot through
    :return: generated obstacles as an x coordinate list + a y coordinate list + a radius list (for all: `r = 0`)
    """
    map_min, map_max = map_range[0], map_range[1]
    map_range_len = map_max - map_min   # l
    n_gaps = math.ceil(map_range_len / (2 * shrink_factor * robot_radius))
    d_real = map_range_len / n_gaps
    n_obstacles = n_gaps + 1

    obstacle_x_list = []
    obstacle_y_list = []

    # bottem edge
    for i in range(n_obstacles):
        obstacle_x_list.append(map_min + d_real * i)
        obstacle_y_list.append(map_min)

    # left edge: bottom one already added, so skip it
    for i in range(1, n_obstacles):
        obstacle_x_list.append(map_min)
        obstacle_y_list.append(map_min + d_real * i)

    # right edge: bottom one already added, so skip it
    for i in range(1, n_obstacles):
        obstacle_x_list.append(map_max)
        obstacle_y_list.append(map_min + d_real * i)

    # top edge: left & right ones already added, so skip both of them
    for i in range(1, n_obstacles - 1):
        obstacle_x_list.append(map_min + d_real * i)
        obstacle_y_list.append(map_max)

    return obstacle_x_list, obstacle_y_list, [0 for _ in range(len(obstacle_x_list))]


def get_test_problem():
    cur_map_range = [0, 60]
    cur_robot_radius = 2
    cur_seed = None
    cur_seed = 666666
    random.seed(cur_seed)
    max_or = 6

    # obstacles
    mox = []
    moy = []
    mor = []
    for i in range(40):
        mox.append(20.0)
        moy.append(i)
        mor.append(random.uniform(0, max_or))
    for i in range(40):
        mox.append(40.0)
        moy.append(60.0 - i)
        mor.append(random.uniform(0, max_or))
    mrx, mry, mrr = gen_map_edge_obstacles(map_range=cur_map_range, robot_radius=cur_robot_radius)
    ox = mox + mrx
    oy = moy + mry
    ors = mor + mrr

    return {
        'map_range': cur_map_range,
        'obstacle_xs': ox,
        'obstacle_ys': oy,
        'obstacle_rs': ors,
        'robot_radius': cur_robot_radius,
        # DEBUG
        'rnd_seed': cur_seed,
        # 'n_samples': 100
    }, {
        'start': [10, 10],
        'goal': [50, 50],
    }


def test_prm(show_map=False, animation=True):
    logger.info('Start PRM program.')

    # Define the problem
    test_problem, test_query = get_test_problem()

    # Create the RRT solver
    logger.info('Initializing the PRM solver...')
    t0 = time.time()
    my_prm = Prm(**test_problem)
    logger.info('PRM setup uses %f sec.' % (time.time() - t0))

    # print map
    if show_map:
        my_prm.draw_graph()
        plt.show()

    # Solve the problem
    logger.info('Start planning...')
    test_query['animation'] = animation
    t1 = time.time()
    my_path, my_path_cost = my_prm.plan(**test_query)
    logger.info('Finish planning using %f sec.' % (time.time() - t1))

    # Timer info
    logger.info('Global PRM timer:\n%s' % my_prm.global_timer)
    logger.info('Query timer:\n%s' % my_prm.query_timer)

    if my_path is None:
        logger.info('Cannot find path.')
        # see what happens
        if animation:
            plt.show()
    else:
        logger.info('Found path! Path cost = %.4f' % my_path_cost)
        logger.info('Path is as follows:\n%s' % my_path)

        # draw path
        if animation:
            my_prm.draw_graph(start=test_query['start'], goal=test_query['goal'],
                              road_map=my_prm.road_map, path=my_path)
            plt.show()


if __name__ == '__main__':
    test_prm()
