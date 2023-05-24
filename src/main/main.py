import os
import pickle
import sys
sys.path.append(os.path.join(os.getcwd(), '../'))   # load core
import argparse
import json
from typing import Optional

from util import Logger
from core.prm import RoadMap
from core.util.graph import RoundObstacle

DATA_DIR_DEFAULT = 'data/'
LOG_DIR_DEFAULT = 'log/'
TAG_DEFAULT = None

algorithm_dict = {
    'PRM': 'Naive-PRM',
    'R-PRM': 'Repair-PRM',
    'RF-PRM': 'Repair-PRM-with-Freedom',
    # 'RFT-PRM': 'Thrifty-Repair-PRM-with-Freedom',
}


def log_horizontal_split(logr: Logger, l: int = 20):
    logr.log('%s\n' % ('-'*l))


def list_data_items(data_folder: str = DATA_DIR_DEFAULT) -> list[dict]:
    """
    Reads the json file from the data folder and retrieves the list of test cases.
    :param data_folder: data folder to be examined
    :return: the test case list, specifying the metadata of each test case
    """
    json_fn = os.path.join(data_folder, 'obstacle_addition_test_cases.json')
    with open(json_fn) as f:
        cases_json = json.load(f)
    return cases_json['test_cases']


def load_add_obstacle_case(test_case: dict) -> (dict, dict, RoadMap, RoundObstacle):
    """
    Loads one test case about obstacle addition.
    :param test_case: the test case dict which stores the case metadata
    :return: a map dict + a query dict + a fixed roadmap + an obstacle to be added
    """
    problem_folder = test_case['map']
    query_ind = test_case['query']
    case_folder = test_case['case']

    # load map args
    with open(os.path.join(problem_folder, 'map.pickle'), 'rb') as f:
        case_map = pickle.load(f)

    # load query args
    with open(os.path.join(problem_folder, 'queries.pickle'), 'rb') as f:
        case_queries = pickle.load(f)
    case_query = case_queries[query_ind]

    # load fixed roadmap
    with open(os.path.join(case_folder, 'roadmap.pickle'), 'rb') as f:
        case_roadmap = pickle.load(f)

    # load new obstacle
    with open(os.path.join(case_folder, 'obstacle.pickle'), 'rb') as f:
        case_obstacle = pickle.load(f)

    return case_map, case_query, case_roadmap, case_obstacle


def run_test_case(case_map: dict, case_query: dict, case_rmp: RoadMap, case_o: RoundObstacle) -> dict:
    """
    Runs the test case and return recorded metrics.
    :param case_map: map for the case
    :param case_query: query for the case
    :param case_rmp: roadmap for the case
    :param case_o: new obstacle for the case
    :return: the recorded metrics for the test case
    """
    metric_res = {
        {
            'algorithm': alg_label,
            'algorithm_label': alg_description,
            'path_result': {
                'path_found': False,
                'path': [],
                'path_cost': -1
            },
            'planning_time': -1,
            'extra': {
                'n_new_nodes': -1,
                'repair_time': -1
            }
        } for (alg_label, alg_description) in algorithm_dict
    }


def main(data_folder: str = DATA_DIR_DEFAULT, logr: Optional[Logger] = None):
    if logr is None:
        logr = Logger(activate=False)

    cases_json = list_data_items(data_folder=data_folder)
    logr.log(f'> {len(cases_json)} cases collected.')

    logr.log('> Start running test cases.')
    for cur_case_json in cases_json:
        log_horizontal_split(logr)
        logr.log('%s\n' % cur_case_json)
        cur_map, cur_query, cur_roadmap, cur_o = load_add_obstacle_case(test_case=cur_case_json)
        # TODO: TEST


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-folder', type=str, default=DATA_DIR_DEFAULT,
                        help=f'where the generated problems are stored, default = {DATA_DIR_DEFAULT}')
    parser.add_argument('-ld', '--log-dir', type=str, default=LOG_DIR_DEFAULT,
                        help=f'Specify where to create a log file if needed, default = {LOG_DIR_DEFAULT}')
    parser.add_argument('-tag', '--tag', type=str, default=TAG_DEFAULT,
                        help=f'Name tag for the model, default = {TAG_DEFAULT}')
    ARGS, _ = parser.parse_known_args()

    ARGS = vars(ARGS)

    logger = Logger(activate=True, logging_folder=ARGS['log_dir'], comment=ARGS['tag']) if ARGS['log_dir'] else Logger(activate=False)
    logger.log('> Main problem initialized.\n')
    logger.log(f'> Input variables: {ARGS}\n')

    main(data_folder=ARGS['data_folder'], logr=logger)

    logger.close()
