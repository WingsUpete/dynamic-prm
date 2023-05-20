import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../'))  # load core

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info('Initializing...')

import numpy as np
import matplotlib.pyplot as plt

from core.rrt import Rrt, RrtNode


def get_test_problem():
    return {
        'map_range': [-2, 15],
        # obstacles as circles (x, y, r)
        'obstacles': [
            (5, 5, 1),
            (3, 6, 2),
            (3, 8, 2),
            (3, 10, 2),
            (7, 5, 2),
            (9, 5, 2),
            (8, 10, 1)
        ],
        'robot_radius': 0.8,
        'start': [0, 0],
        'goal': [6, 10],
        # 'max_iter': 500
        # DEBUG
        # 'rnd_seed': 66666,
        # 'pause_interval': 3
    }


def test_steering():
    test_problem = get_test_problem()
    test_problem['expand_dis'] = 8
    test_problem['path_resolution'] = 3

    rrt = Rrt(**test_problem)
    from_node = RrtNode(0, 0)
    to_node = RrtNode(10, 0)

    logger.info('expand_dis = %.4f' % rrt.expand_dis)
    logger.info('path_resolution = %.4f' % rrt.path_resolution)
    logger.info('from %s to %s' % (from_node, to_node))
    new_node = rrt.steer(from_node=from_node, to_node=to_node)
    logger.info('new node: %s' % new_node)
    path = np.vstack((new_node.path_x, new_node.path_y)).T
    logger.info('path:\n%s' % path)


def test_rrt(animation=True):
    logger.info('Start RRT program.')

    # Define the problem
    test_problem = get_test_problem()

    # Create the RRT solver
    logger.info('Initializing the RRT solver...')
    rrt = Rrt(**test_problem)

    # Solve the problem
    logger.info('Start planning...')
    my_path, my_path_cost, used_steps = rrt.plan(animation=animation)
    logger.info('Finish planning.')

    if my_path is None:
        logger.info('Cannot find path within %d steps.' % used_steps)
    else:
        logger.info('Found path using %d steps! Path cost = %.4f' % (used_steps, my_path_cost))
        logger.info('Path is as follows:\n%s' % my_path)

        if animation:
            rrt.draw_graph(path=my_path)
            plt.show()


if __name__ == '__main__':
    # test_steering()
    test_rrt()
