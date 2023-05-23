import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../'))  # load core
import time
import pickle

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info('Initializing...')

import matplotlib.pyplot as plt

from core.prm import *


def test_prm(use_rrt=False, show_map=False, animation=True):
    logger.info('Start PRM program.')

    # For stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None]
    )

    # Load the sample problem
    test_problem_folder = '../../sample_data/test_problem/'
    with open(os.path.join(test_problem_folder, 'map.pickle'), 'rb') as f:
        test_map = pickle.load(f)
    with open(os.path.join(test_problem_folder, 'queries.pickle'), 'rb') as f:
        test_queries = pickle.load(f)
    test_query = test_queries[0]

    logger.info('Seed: %d' % test_map['rnd_seed'])

    # Create the RRT solver
    logger.info('Initializing the PRM solver...')
    test_map['init_n_samples'] = 20
    t0 = time.time()
    my_prm = Prm(**test_map)
    logger.info('PRM setup uses %f sec.' % (time.time() - t0))

    # print map
    if show_map:
        my_prm.draw_graph()
        plt.show()

    # Solve the problem
    logger.info('Start planning...')
    test_query['animation'] = animation
    t1 = time.time()
    rrt_road_map = None
    if use_rrt:
        my_path, my_path_cost, rrt_road_map = my_prm.plan_rrt(**test_query)
    else:
        my_path, my_path_cost = my_prm.plan(**test_query)
    logger.info('Finish planning using %f sec.' % (time.time() - t1))

    # Timer info
    logger.info('Global PRM timer:\n%s' % my_prm.global_timer)
    logger.info('Query timer:\n%s' % my_prm.query_timer)

    if my_path is None:
        logger.info('Cannot find path.')
        # see what happens
        plt.show()
    else:
        logger.info('Found path! Path cost = %.4f' % my_path_cost)
        logger.info('Path is as follows:\n%s' % my_path)

        # draw path
        my_prm.draw_graph(start=test_query['start'], goal=test_query['goal'],
                          road_map=rrt_road_map if use_rrt else my_prm.road_map,
                          path=my_path)
        plt.show()


if __name__ == '__main__':
    test_prm(
        use_rrt=True,
        # show_map=True,
        # animation=False
    )
