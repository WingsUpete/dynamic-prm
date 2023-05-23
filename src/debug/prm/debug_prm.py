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
from core.util.graph import *

# For stopping simulation with the esc key.
plt.gcf().canvas.mpl_connect(
    'key_release_event',
    lambda event: [exit(0) if event.key == 'escape' else None]
)


def load_sample_problem(animation=True):
    # Load the sample problem
    test_problem_folder = '../../sample_data/test_problem/'
    with open(os.path.join(test_problem_folder, 'map.pickle'), 'rb') as f:
        test_map = pickle.load(f)
    with open(os.path.join(test_problem_folder, 'queries.pickle'), 'rb') as f:
        test_queries = pickle.load(f)
    test_query = test_queries[0]
    test_query['animation'] = animation
    logger.info('Seed: %d' % test_map['rnd_seed'])
    return test_map, test_query


def test_prm(use_rrt=False, show_map=False, animation=True):
    logger.info('Start PRM program.')

    # Load the sample problem
    test_map, test_query = load_sample_problem(animation)
    # test_map['init_n_samples'] = 200

    # Create the PRM solver
    logger.info('Initializing the PRM solver...')
    t0 = time.time()
    my_prm = Prm(**test_map)
    logger.info('PRM setup uses %f sec.' % (time.time() - t0))

    # print map
    if show_map:
        my_prm.draw_graph()
        plt.show()

    # Solve the problem
    logger.info('Start planning...')
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


def analyze_path(my_prm: Prm, my_path: list[list[float]], my_path_cost: float, test_query: dict):
    if my_path is None:
        logger.info('Cannot find path.')
        return
    logger.info('Path cost = %.4f' % my_path_cost)
    my_prm.draw_graph(start=test_query['start'], goal=test_query['goal'],
                      road_map=my_prm.road_map, path=my_path)


def test_add_obstacle():
    test_map, test_query = load_sample_problem(animation=True)
    my_prm = Prm(**test_map)
    logger.info('PRM solver constructed.')
    my_prm.draw_graph()
    plt.waitforbuttonpress()

    # query 1st time
    logger.info('1st query.')
    my_path, my_path_cost = my_prm.plan(**test_query)
    analyze_path(my_prm, my_path, my_path_cost, test_query)
    plt.waitforbuttonpress()

    # add obstacle
    add_obstacle = RoundObstacle(x=32, y=22, r=2, obstacle_type=ObstacleType.NORMAL, obstacle_uid='add')
    logger.info('Add one obstacle: %s' % add_obstacle)
    my_prm.add_obstacle_to_environment(obstacle=add_obstacle)
    my_prm.draw_graph(start=test_query['start'], goal=test_query['goal'],
                      road_map=my_prm.road_map, path=my_path)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    test_prm(
        # use_rrt=True,
        # show_map=True,
        # animation=False
    )
    # test_add_obstacle()
