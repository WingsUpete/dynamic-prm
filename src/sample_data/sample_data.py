import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../'))  # load core
import random
from typing import Optional
import pickle

from core.util import ObstacleDict, RoundObstacle, ObstacleType


def get_test_problem(cur_seed: Optional[int] = None, init_n_samples: Optional[int] = None) -> (dict, dict):
    """
    Defines the test problem: map + queries
    :param cur_seed: seed value for the random generator
    :param init_n_samples: initial number of points to sample for PRM
    :return: the test map and queries for the problem, as dicts
    """
    random.seed(cur_seed)

    cur_map_range = [0, 60]
    cur_robot_radius = 2
    max_or = 6

    # obstacles
    o_dict = ObstacleDict(map_range=cur_map_range, robot_radius=cur_robot_radius)
    for i in range(40):
        cur_o = RoundObstacle(x=20.0, y=i, r=random.uniform(0, max_or), obstacle_type=ObstacleType.NORMAL)
        o_dict.add_obstacle(obstacle=cur_o)
    for i in range(40):
        cur_o = RoundObstacle(x=40.0, y=60.0 - i, r=random.uniform(0, max_or), obstacle_type=ObstacleType.NORMAL)
        o_dict.add_obstacle(obstacle=cur_o)

    # MAP
    test_map = {
        'map_range': cur_map_range,
        'obstacles': o_dict,
        'robot_radius': cur_robot_radius,

        'rnd_seed': cur_seed,
    }
    if init_n_samples:
        test_map['init_n_samples'] = init_n_samples

    # QUERY
    test_queries = [
        {
            'start': [10, 10],
            'goal': [50, 50],
        }
    ]

    return test_map, test_queries


def store_test_problem(folder: str = './test_problem') -> None:
    """
    Stores the test problem to local by dumping the problem.
    :param folder: specifies where to store the problem
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)

    test_map, test_queries = get_test_problem(
        # cur_seed=666666,
        # init_n_samples=100
    )

    to_files = [
        (os.path.join(folder, 'map.pickle'), test_map),
        (os.path.join(folder, 'queries.pickle'), test_queries),
    ]

    for (f_name, cur_item) in to_files:
        with open(f_name, 'wb') as f:
            pickle.dump(cur_item, f)


if __name__ == '__main__':
    store_test_problem()
