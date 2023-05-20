import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../'))  # load core

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info('Initializing...')

import matplotlib.pyplot as plt

from core.prm import *


def get_test_problem():
    # obstacles
    ox = []
    oy = []
    for i in range(60):
        ox.append(i)
        oy.append(0.0)
    for i in range(60):
        ox.append(60.0)
        oy.append(i)
    for i in range(61):
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(40):
        ox.append(40.0)
        oy.append(60.0 - i)

    return {
        # 'map_range': [-10, 70],
        'map_range': [0, 60],
        'obstacle_xs': ox,
        'obstacle_ys': oy,
        'robot_radius': 5.0,
        # DEBUG
        # 'rnd_seed': 66666,
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
    my_prm = Prm(**test_problem)

    # print map
    if show_map:
        my_prm.draw_graph()
        plt.show()

    # Solve the problem
    logger.info('Start planning...')
    test_query['animation'] = animation
    my_path, my_path_cost = my_prm.plan(**test_query)
    logger.info('Finish planning.')

    if my_path is None:
        logger.info('Cannot find path.')
    else:
        logger.info('Found path! Path cost = %.4f' % my_path_cost)
        logger.info('Path is as follows:\n%s' % my_path)

        # draw path
        if animation:
            my_prm.draw_graph(start=test_query['start'], goal=test_query['goal'],
                              sample_point_x_list=my_prm.sample_x, sample_point_y_list=my_prm.sample_y,
                              road_map=my_prm.road_map, path=my_path)
            plt.show()


if __name__ == '__main__':
    test_prm()
