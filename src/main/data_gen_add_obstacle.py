import math
import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../'))  # load core
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info('Initializing...')
import argparse
from typing import Optional
import random
import json
import pickle
import copy

from tqdm import tqdm
import matplotlib.pyplot as plt

from core.util.common import *
from core.util.graph import ObstacleType, RoundObstacle
from core.prm import Prm, RoadMapNode, RoadMap

DATA_DIR_DEFAULT = './data/'
SEED_DEFAULT = None
N_CASES_DEFAULT = 100

init_n_samples_list = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300]
n_init_prm_attempts = 10    # for each `init_n_samples`, construct road map 10 times to get a feasible one
n_add_obstacle_attempt = 3  # sample one nice obstacle to add, sample at most 3 times for a query
n_single_case_test = 3  # use RRT to test each case 3 times (need to succeed in connection in all tests)
robot_radius = 1
size_obstacle_to_robot_ratio = [0.5, 5]  # 0.5 < r_obstacle : r_robot <= 5
min_or = size_obstacle_to_robot_ratio[0] * robot_radius
max_or = size_obstacle_to_robot_ratio[1] * robot_radius

img_dpi = 200


def try_add_obstacle(cur_map: dict, cur_query: dict) -> (Optional[RoadMap], Optional[RoundObstacle]):
    """TODO"""
    # guaranteed to find a path using PRM
    cur_prm = None
    cur_path = None
    for init_n_samples in init_n_samples_list:
        for _ in range(n_init_prm_attempts):
            cur_prm = Prm(init_n_samples=init_n_samples, **cur_map)
            cur_path, _ = cur_prm.plan(animation=False, **cur_query)
            if cur_path:
                break
        if cur_path:
            break

    if cur_path is None:
        return None, None

    cur_prm_backup = copy.deepcopy(cur_prm)

    # first and final node should be valid
    selectable_nodes = cur_path[1:len(cur_path) - 1]

    # number of sample attempts = 3 * (num_selectable_nodes - 1)
    n_sample_attempt = 2 * (len(selectable_nodes) - 1)
    for _ in range(n_sample_attempt):
        # select one consecutive node pair (cur -> next) to block
        cur_ind = random.randint(0, len(selectable_nodes) - 2)
        cur_node = selectable_nodes[cur_ind]
        nxt_node = selectable_nodes[cur_ind + 1]

        # sample obstacle radius
        sample_or = random.uniform(min_or, max_or)

        d, theta = cal_dist_n_angle(from_x=cur_node[0], from_y=cur_node[1], to_x=nxt_node[0], to_y=nxt_node[1])
        # sample where the obstacle will be placed
        sample_d = random.uniform(0, d)
        sample_ox = cur_node[0] + sample_d * math.cos(theta)
        sample_oy = cur_node[1] + sample_d * math.sin(theta)

        # construct obstacle
        sample_o = RoundObstacle(x=sample_ox, y=sample_oy, r=sample_or, obstacle_type=ObstacleType.NORMAL)

        # add obstacle to the environment
        cur_prm = copy.deepcopy(cur_prm_backup)
        cur_prm.add_obstacle_to_environment(obstacle=sample_o)

        # now PRM should fail; if not, skip
        cur_path, _ = cur_prm.plan(animation=False, **cur_query)
        if cur_path:
            continue

        # now PRM fails, but RRT should steadily work with the query
        rrt_works = True
        for _ in range(n_single_case_test):
            rrt_path, _, _ = cur_prm.plan_rrt(animation=False, **cur_query)
            if rrt_path is None:
                # all tests must pass, to have a strong case
                rrt_works = False
                break

        if rrt_works:
            # nice case
            return cur_prm_backup.road_map, sample_o

    # Not found
    return None, None


def handle_problem(problem_folder: str) -> list[dict]:
    """TODO"""
    with open(os.path.join(problem_folder, 'map.pickle'), 'rb') as f:
        test_map = pickle.load(f)
    with open(os.path.join(problem_folder, 'queries.pickle'), 'rb') as f:
        test_queries = pickle.load(f)

    gen_cases = []
    for i in range(len(test_queries)):
        test_query = test_queries[i]
        for _ in range(n_add_obstacle_attempt):
            new_roadmap, new_obstacle = try_add_obstacle(cur_map=test_map, cur_query=test_query)
            if new_roadmap and new_obstacle:
                case_folder = os.path.join(problem_folder, f'case_{new_obstacle.obstacle_uid}/')
                os.mkdir(case_folder)

                # Saves roadmap and obstacle
                rmp_fn = os.path.join(case_folder, f'roadmap.pickle')
                with open(rmp_fn, 'wb') as f:
                    pickle.dump(new_roadmap, f)
                o_fn = os.path.join(case_folder, f'obstacle.pickle')
                with open(o_fn, 'wb') as f:
                    pickle.dump(new_obstacle, f)

                # Records case metadata
                gen_cases.append({
                    'map': problem_folder,
                    'query': i,
                    'case': case_folder,
                })

                # Saves figures about cases
                cur_prm = Prm(roadmap=copy.deepcopy(new_roadmap), **test_map)
                prm_path, _ = cur_prm.plan(animation=False, **test_query)
                cur_prm.add_obstacle_to_environment(obstacle=new_obstacle)
                cur_prm.draw_graph(start=test_query['start'], goal=test_query['goal'],
                                   road_map=cur_prm.road_map, path=prm_path, pause=False)
                plt.savefig(os.path.join(case_folder, 'prm.png'), dpi=img_dpi)

                rrt_path, _, rrt_roadmap = cur_prm.plan_rrt(animation=False, **test_query)
                cur_prm.draw_graph(start=test_query['start'], goal=test_query['goal'],
                                   road_map=rrt_roadmap, path=rrt_path, pause=False)
                plt.savefig(os.path.join(case_folder, 'rrt.png'), dpi=img_dpi)

    return gen_cases


def gen_obstacle_addition_cases(output_folder: str = DATA_DIR_DEFAULT,
                                n_cases: int = N_CASES_DEFAULT,
                                seed: Optional[int] = SEED_DEFAULT):
    """TODO"""
    random.seed(seed)
    logger.info(f'Random generator seeded to {seed}')

    assert os.path.isdir(output_folder)

    summary_fn = os.path.join(output_folder, 'obstacle_addition_test_cases.json')
    if os.path.isfile(summary_fn):
        with open(summary_fn) as f:
            test_cases = json.load(f)['test_cases']
    else:
        # {"map": "data/12", "query": 6, "case": "data/12/case_obstacle_uid/"}
        test_cases = []

    logger.info('Starting generating test cases...')
    dirs = os.listdir(output_folder)
    random.shuffle(dirs)
    logger.info('folders: %s' % dirs)
    pbar = tqdm(total=n_cases)
    pbar.update(len(test_cases))
    for item in dirs:
        cur_folder = os.path.join(output_folder, item)
        if not os.path.isdir(cur_folder):
            continue
        pbar.set_description(cur_folder, refresh=True)
        new_cases = handle_problem(problem_folder=cur_folder)
        test_cases += new_cases
        pbar.update(len(new_cases))
        if len(test_cases) > n_cases:
            test_cases = test_cases[:n_cases]
            break

        summary = {
            'folder': output_folder,
            'n_cases': len(test_cases),
            'test_cases': test_cases
        }
        with open(summary_fn, 'w') as f:
            json.dump(summary, f)
    pbar.close()
    logger.info('%d test cases generated.' % len(test_cases))

    summary = {
        'folder': output_folder,
        'n_cases': len(test_cases),
        'test_cases': test_cases
    }
    with open(summary_fn, 'w') as f:
        json.dump(summary, f)
    logger.info(f'Saved "{summary_fn}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates test problems for experiment')

    parser.add_argument('-o', '--output-folder', type=str, default=DATA_DIR_DEFAULT,
                        help=f'where the generated problems are stored, default = {DATA_DIR_DEFAULT}')
    parser.add_argument('-n', '--n-cases', type=int, default=N_CASES_DEFAULT,
                        help=f'how many test cases to generate, default = {N_CASES_DEFAULT}')
    parser.add_argument('-s', '--seed', type=Optional[int], default=SEED_DEFAULT,
                        help=f'seed value for the random generator, default = {SEED_DEFAULT}')

    ARGS, _ = parser.parse_known_args()

    ARGS = vars(ARGS)

    logger.info(f'Inputs: {ARGS}')

    gen_obstacle_addition_cases(**ARGS)
