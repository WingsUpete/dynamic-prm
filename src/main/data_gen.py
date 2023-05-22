import os
import sys

sys.path.append(os.path.join(os.getcwd(), '../'))  # load core
from typing import Optional
import logging
import pickle
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info('Initializing...')
import argparse
import math

from tqdm import tqdm
import matplotlib.pyplot as plt

from core.util.graph import ObstacleType, ObstacleDict, RoundObstacle
from core.prm import Prm

DATA_DIR_DEFAULT = './data/'
N_PROB_DEFAULT = 100
N_QUERIES_DEFAULT = 3
SEED_DEFAULT = None
ROBOT_RADIUS_DEFAULT = 2

map_len_range = [50, 500]
size_obstacle_to_robot_ratio = 2  # r_obstacle : r_robot = 2
obstacle_coverage = 0.2  # obstacles cover at most 20% of the map
max_n_sample_point_attempt = 20  # sample at most 20 starting/goal points, until finding one feasible point
n_single_query_test = 3  # use PRM to test each query 3 times (need to succeed in all tests)
query_success_rate_threshold = 0.8  # 80% of the sampled queries should have feasible paths; otherwise discard this map

img_dpi = 200


def store_problem(problem_folder: str, problem_map: dict, problem_queries: list[dict]) -> None:
    """
    Stores the problem to local by dumping the problem.
    :param problem_folder: the folder holding all problem-related data
    :param problem_map: map as a dict for the problem
    :param problem_queries: list of queries as a list of dicts
    """
    if not os.path.isdir(problem_folder):
        os.mkdir(problem_folder)

    # (destination_file, item)
    store_tasks = [
        (os.path.join(problem_folder, 'map.pickle'), problem_map),
        (os.path.join(problem_folder, 'queries.pickle'), problem_queries),
    ]

    for (f_name, cur_item) in store_tasks:
        with open(f_name, 'wb') as f:
            pickle.dump(cur_item, f)

    # save a map figure
    o_dict: ObstacleDict = problem_map['obstacles']
    o_dict.draw_map_edge_n_obstacles()
    plt.savefig(os.path.join(problem_folder, 'map.png'), dpi=img_dpi)

    # save queries figures
    query_fig_folder = os.path.join(problem_folder, 'query_fig/')
    if not os.path.isdir(query_fig_folder):
        os.mkdir(query_fig_folder)
    for q_id in range(len(problem_queries)):
        cur_prm = Prm(**problem_map)
        cur_path, _ = cur_prm.plan(animation=False, **problem_queries[q_id])
        cur_prm.draw_graph(start=problem_queries[q_id]['start'], goal=problem_queries[q_id]['goal'],
                           road_map=cur_prm.road_map, path=cur_path, pause=False)
        plt.savefig(os.path.join(query_fig_folder, f'{q_id}.png'), dpi=img_dpi)


def sample_feasible_point(o_dict: ObstacleDict, map_range: list[float]) -> Optional[list[float]]:
    """
    Samples a feasible point on the map that does not collide with any obstacle.
    :param o_dict: provided dictionary of obstacles
    :param map_range: [min, max] of the map
    :return: a feasible point, or None if not found after maximum number of attempts
    """
    for _ in range(max_n_sample_point_attempt):
        cur_x = random.uniform(*map_range)
        cur_y = random.uniform(*map_range)
        if not o_dict.point_collides(x=cur_x, y=cur_y):
            return [cur_x, cur_y]

    return None


def gen_problem(robot_radius: float = ROBOT_RADIUS_DEFAULT, n_queries: int = N_QUERIES_DEFAULT) -> (dict, list[dict]):
    """
    Generates a problem for the robot. More details:

    1. Map range: min is 0, uniformly sample max between 20 and 100. Then, uniformly sample the actual map length
    between min and max.

    2. Maximum allowed obstacle radius: 2 times as large as the robot.

    3. Number of obstacles: in total, obstacles occupy 20% of the map at most:
    avg_or = max_or / 2 -> n * pi * avg_or^2 / L^2 = 0.2, L is map length. In the most extreme case, we still
    want at least 1 obstacle on the map.

    4. Start/Goal point for each query: uniformly sample on the map, until it is infeasible (collides with certain
    obstacle). If feasible points cannot be found after maximum attempts, discard the whole map. If there is no obstacle
    on the way between the two points, then also discard this point pair (since it is not a good sample)

    5. Query: each query is tested with PRM several times and is feasible if all tests manage to find a path. If the
    success rate hits lower than a pre-set threshold, discard the whole map.
    :param robot_radius: radius of the round
    :param n_queries: number of queries to generate per problem
    :return: problem_map as dict and list of queries as a list of dicts
    """
    # sample the side length of the map
    map_len = random.randint(*map_len_range)
    map_range = [0, float(map_len)]

    # maximum allowed obstacle radius
    max_or = size_obstacle_to_robot_ratio * robot_radius

    # number of obstacles: avg_or = max_or / 2 -> n * pi * avg_or^2 / L^2 = 0.2, L is map length
    n_obstacles: int = math.floor(obstacle_coverage * map_len ** 2 / (math.pi * (max_or / 2) ** 2))
    n_obstacles = 1 if n_obstacles == 0 else n_obstacles  # at least 1 obstacle

    while True:
        # initialize obstacle dict and generate map edge point obstacles
        o_dict = ObstacleDict(map_range=map_range, robot_radius=robot_radius)
        # generate normal obstacles
        for i in range(n_obstacles):
            cur_ox = random.uniform(*map_range)
            cur_oy = random.uniform(*map_range)
            cur_or = random.uniform(0, max_or)
            cur_o = RoundObstacle(x=cur_ox, y=cur_oy, r=cur_or, obstacle_type=ObstacleType.NORMAL)
            o_dict.add_obstacle(obstacle=cur_o)

        # construct problem map
        problem_map = {
            'map_range': map_range,
            'obstacles': o_dict,
            'robot_radius': robot_radius,
        }

        # generate queries
        problem_queries = []
        # it requires that `n / N >= t`, where `n` is `n_queries`, `N` is the actual number of sampled queries, and
        # `t` is the success rate threshold. Therefore, `N = floor(n / t)`
        max_n_sample_queries: int = math.floor(n_queries / query_success_rate_threshold)
        for _ in range(max_n_sample_queries):
            while True:
                cur_start = sample_feasible_point(o_dict=o_dict, map_range=map_range)
                cur_goal = sample_feasible_point(o_dict=o_dict, map_range=map_range)
                if (cur_start is None) or (cur_goal is None):
                    break
                # if no obstacle lies between the two points, this sample query is not good enough.
                if not o_dict.reachable_without_collision(from_x=cur_start[0], from_y=cur_start[1],
                                                          to_x=cur_goal[0], to_y=cur_goal[1]):
                    break

            # cannot find feasible start-goal point pair: discard this map
            if (cur_start is None) or (cur_goal is None):
                break

            cur_problem_query = {
                'start': cur_start,
                'goal': cur_goal,
            }

            query_feasible = True
            for _ in range(n_single_query_test):
                cur_prm = Prm(**problem_map)
                cur_path, _ = cur_prm.plan(animation=False, **cur_problem_query)
                if cur_path is None:
                    # all query tests must pass, to have a strong sampled case.
                    query_feasible = False
                    break

            if query_feasible:
                problem_queries.append(cur_problem_query)
                if len(problem_queries) == n_queries:
                    break

        if len(problem_queries) < n_queries:
            # not enough queries, this map is not good.
            continue

        # otherwise, enough feasible queries, good map, return this problem
        return problem_map, problem_queries


def gen_problems_to_local(output_folder: str = DATA_DIR_DEFAULT,
                          n_problems: int = N_PROB_DEFAULT, n_queries: int = N_QUERIES_DEFAULT,
                          robot_radius: float = ROBOT_RADIUS_DEFAULT, seed: Optional[int] = SEED_DEFAULT) -> None:
    """
    Generates multiple problems and store them to local disk.
    :param output_folder: the folder to store the problems
    :param n_problems: number of problems to generate
    :param n_queries: number of queries to generate per problem
    :param robot_radius: radius of the round robot
    :param seed: seed for the random generator
    """
    random.seed(seed)
    logger.info(f'Random generator seeded to {seed}')

    if not os.path.isdir(output_folder):
        logger.info(f'"{output_folder}" does not exist. Create it.')
        os.mkdir(output_folder)

    logger.info('Start generating problems...')
    for i in tqdm(range(n_problems)):
        cur_sample_folder = os.path.join(output_folder, str(i))
        cur_map, cur_queries = gen_problem(robot_radius=robot_radius, n_queries=n_queries)
        store_problem(problem_folder=cur_sample_folder, problem_map=cur_map, problem_queries=cur_queries)

    logger.info('%d problems generated.' % n_problems)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates test problems for experiment')

    parser.add_argument('-o', '--output-folder', type=str, default=DATA_DIR_DEFAULT,
                        help=f'where the generated problems are stored, default = {DATA_DIR_DEFAULT}')
    parser.add_argument('-np', '--n-problems', type=int, default=N_PROB_DEFAULT,
                        help=f'how many problems to generate, default = {N_PROB_DEFAULT}')
    parser.add_argument('-nq', '--n-queries', type=int, default=N_QUERIES_DEFAULT,
                        help=f'how many queries to generate problem-wise, default = {N_QUERIES_DEFAULT}')
    parser.add_argument('-r', '--robot-radius', type=float, default=ROBOT_RADIUS_DEFAULT,
                        help=f'radius of the round robot, default = {ROBOT_RADIUS_DEFAULT}')
    parser.add_argument('-s', '--seed', type=Optional[int], default=SEED_DEFAULT,
                        help=f'seed value for the random generator, default = {SEED_DEFAULT}')

    ARGS, _ = parser.parse_known_args()

    ARGS = vars(ARGS)

    logger.info(f'Inputs: {ARGS}')

    gen_problems_to_local(**ARGS)
