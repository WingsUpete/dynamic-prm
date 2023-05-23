import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../'))  # load core
import random
from typing import Optional
import pickle

import matplotlib.pyplot as plt

from core.util import ObstacleDict, RoundObstacle, ObstacleType
from core.prm import Prm

img_dpi = 300


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
        cur_o = RoundObstacle(x=20.0, y=i, r=random.uniform(0, max_or),
                              obstacle_uid=f'l{i}', obstacle_type=ObstacleType.NORMAL)
        o_dict.add_obstacle(obstacle=cur_o)
    for i in range(40):
        cur_o = RoundObstacle(x=40.0, y=60.0 - i, r=random.uniform(0, max_or),
                              obstacle_uid=f'r{i}', obstacle_type=ObstacleType.NORMAL)
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


def store_test_problem(folder: str = './test_problem/') -> None:
    """
    Stores the test problem to local by dumping the problem.
    :param folder: specifies where to store the problem
    """
    if not os.path.isdir(folder):
        os.mkdir(folder)

    test_map, test_queries = get_test_problem(
        cur_seed=666666,
        init_n_samples=20
    )

    to_files = [
        (os.path.join(folder, 'map.pickle'), test_map),
        (os.path.join(folder, 'queries.pickle'), test_queries),
    ]

    for (f_name, cur_item) in to_files:
        with open(f_name, 'wb') as f:
            pickle.dump(cur_item, f)

    # save a map figure
    o_dict: ObstacleDict = test_map['obstacles']
    o_dict.draw_map_edge_n_obstacles()
    plt.savefig(os.path.join(folder, 'map.png'), dpi=img_dpi)

    # save queries figures
    for q_id in range(len(test_queries)):
        cur_prm = Prm(**test_map)
        cur_path, _ = cur_prm.plan(animation=False, **test_queries[q_id])
        cur_prm.draw_graph(start=test_queries[q_id]['start'], goal=test_queries[q_id]['goal'],
                           road_map=cur_prm.road_map, path=cur_path, pause=False)
        plt.savefig(os.path.join(folder, f'query_{q_id}.png'), dpi=img_dpi)


if __name__ == '__main__':
    store_test_problem()
