import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../'))  # load core
import random
import time

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info('Initializing...')

import matplotlib.pyplot as plt

from core.prm import *
from core.util import *


def get_test_problem():
    cur_map_range = [0, 60]
    cur_robot_radius = 2
    cur_seed = None
    # cur_seed = 666666
    random.seed(cur_seed)
    max_or = 6

    # obstacles
    o_dict = ObstacleDict(map_range=cur_map_range, robot_radius=cur_robot_radius)
    for i in range(40):
        cur_o = RoundObstacle(x=20.0, y=i, r=random.uniform(0, max_or), obstacle_type=ObstacleType.NORMAL)
        o_dict.add_obstacle(obstacle=cur_o)
    for i in range(40):
        cur_o = RoundObstacle(x=40.0, y=60.0 - i, r=random.uniform(0, max_or), obstacle_type=ObstacleType.NORMAL)
        o_dict.add_obstacle(obstacle=cur_o)

    return {
        'map_range': cur_map_range,
        'obstacles': o_dict,
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
