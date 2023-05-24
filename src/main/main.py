import os
import pickle
import sys
sys.path.append(os.path.join(os.getcwd(), '../'))   # load core
import argparse
import json
from typing import Optional
from pprint import pformat
from collections import OrderedDict

from util import Logger
from core.prm import RoadMap, Prm
from core.util.graph import RoundObstacle

DATA_DIR_DEFAULT = 'data/'
LOG_DIR_DEFAULT = 'log/'
TAG_DEFAULT = None
N_RUNS_DEFAULT = 20

algorithm_dict = OrderedDict([
    ('PRM', 'Naive-PRM'),
    ('RRT', 'Naive-RRT'),
    ('R-PRM', 'Repair-PRM'),
    ('RF-PRM', 'Repair-PRM-with-Freedom'),
    # ('RFT-PRM', 'Thrifty-Repair-PRM-with-Freedom'),
])


def log_horizontal_split(logr: Logger, ll: int = 80) -> None:
    logr.log('%s\n' % ('-'*ll))


def log_result(res: dict, logr: Logger) -> None:
    """
    Format logs the result metric dict
    :param res: res dict to log
    :param logr: logger to perform the logging
    """
    logr.log('algorithm\t\tpath_found\t\tpath_cost\t\tplanning_time\t\trepair_time\t\tn_new_nodes\n')
    for alg_label in algorithm_dict.keys():
        logr.log('%-10s\t\t%.2f\t\t\t%-9s\t\t\t%.4f\t\t\t\t%.4f\t\t\t%.2f\n' % (
            alg_label, res[alg_label]['path_result']['path_found'], '%.4f' % res[alg_label]['path_result']['path_cost'],
            res[alg_label]['planning_time'],
            res[alg_label]['extra']['repair_time'], res[alg_label]['extra']['n_new_nodes']
        ))
    logr.log('\n')


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


def record_path_res(metric_dict: dict, path: Optional[list[list[float]]], cost: float) -> dict:
    """
    Records path results to the given metric dict, according to the path + cost result.
    :param metric_dict: given metric dict to be updated
    :param path: resulted path
    :param cost: resulted cost
    :return: the updated metric dict
    """
    if path:
        metric_dict['path_result']['path_found'] = 1
        metric_dict['path_result']['path'] = path
        metric_dict['path_result']['path_cost'] = cost

    return metric_dict


def construct_prm_solver_with_obstacle(case_map: dict, case_rmp: RoadMap, case_o: RoundObstacle) -> Prm:
    """
    Constructs a new PRM solver with the new obstacle added.
    :param case_map: map for the case
    :param case_rmp: roadmap for the case
    :param case_o: new obstacle for the case
    :return: the constructed PRM solver
    """
    case_prm = Prm(roadmap=case_rmp, **case_map)
    case_prm.add_obstacle_to_environment(obstacle=case_o)
    return case_prm


def construct_metric_res(aggregate: bool = False) -> dict:
    """
    Constructs a metric result dict for a single run, or aggregated version for multiple runs.
    :param aggregate: specifies whether to construct the aggregated version
    :return: the constructed metric result dict
    """
    metric_res = {
        alg_label: {
            'algorithm': alg_label,
            'algorithm_label': alg_description,
            'path_result': {
                'path_found': 0,
                'path': None if aggregate else [],
                'path_cost': 0
            },
            'planning_time': 0,
            'extra': {
                'repair_time': 0,
                'n_new_nodes': 0,
            }
        } for (alg_label, alg_description) in algorithm_dict.items()
    }
    return metric_res


def aggregate_metric_res(aggr_metric_dict: dict, cur_metric_res: dict) -> dict:
    """
    Aggregates the current metric result into the aggregate metric dict
    :param aggr_metric_dict: the aggregate metric dict to be updated
    :param cur_metric_res: current metric res for updating
    :return: the updated aggregate metric dict
    """
    for alg_label in algorithm_dict.keys():
        # path
        aggr_metric_dict[alg_label]['path_result']['path_found'] += cur_metric_res[alg_label]['path_result']['path_found']
        aggr_metric_dict[alg_label]['path_result']['path_cost'] += cur_metric_res[alg_label]['path_result']['path_cost']
        # others
        aggr_metric_dict[alg_label]['planning_time'] += cur_metric_res[alg_label]['planning_time']
        aggr_metric_dict[alg_label]['extra']['n_new_nodes'] += cur_metric_res[alg_label]['extra']['n_new_nodes']
        aggr_metric_dict[alg_label]['extra']['repair_time'] += cur_metric_res[alg_label]['extra']['repair_time']

    return aggr_metric_dict


def cal_relative_delta(cur: float, base: float) -> float:
    """
    Calculates the relative delta of current result compared to the base value.
    :param cur: current result
    :param base: base value
    :return: calculated relative delta result
    """
    return (cur - base) / (abs(base) + 1)


def postproc_aggr_metric_dict(aggr_metric_dict: dict, n_runs: int,
                              handle_path_cost: bool = True, calc_delta: bool = True) -> (dict, Optional[dict]):
    """
    Postprocesses the aggregate metric dict. Also calculates a delta metric dict
    :param aggr_metric_dict: the aggregate metric dict to be postprocessed
    :param n_runs: number of runs during the aggregation
    :param handle_path_cost: specifies whether to average path cost by number of found paths (if not, average by `n_runs`)
    :param calc_delta: specifies whether to calculate the delta metric dict
    :return: the postprocessed aggregate metric result + calculated delta metric result
    """
    for alg_label in algorithm_dict.keys():
        # path
        if handle_path_cost:
            if aggr_metric_dict[alg_label]['path_result']['path_found'] > 0:
                aggr_metric_dict[alg_label]['path_result']['path_cost'] /= aggr_metric_dict[alg_label]['path_result']['path_found']
                aggr_metric_dict[alg_label]['path_result']['path_found'] /= n_runs
        else:
            aggr_metric_dict[alg_label]['path_result']['path_cost'] /= n_runs
            aggr_metric_dict[alg_label]['path_result']['path_found'] /= n_runs
        # others
        aggr_metric_dict[alg_label]['planning_time'] /= n_runs
        aggr_metric_dict[alg_label]['extra']['n_new_nodes'] /= n_runs
        aggr_metric_dict[alg_label]['extra']['repair_time'] /= n_runs

    if not calc_delta:
        return aggr_metric_dict, None

    # Compared to Naive PRM
    delta_metric_dict = construct_metric_res(aggregate=True)
    for alg_label in algorithm_dict.keys():
        if alg_label == 'PRM':
            continue
        for nested_metric in ['path_result', 'extra']:
            for inner_metric in delta_metric_dict[alg_label][nested_metric].keys():
                if (nested_metric == 'path_result') and (inner_metric == 'path'):
                    continue
                delta_metric_dict[alg_label][nested_metric][inner_metric] = cal_relative_delta(
                    cur=aggr_metric_dict[alg_label][nested_metric][inner_metric],
                    base=aggr_metric_dict['PRM'][nested_metric][inner_metric]
                )
        delta_metric_dict[alg_label]['planning_time'] = cal_relative_delta(
            cur=aggr_metric_dict[alg_label]['planning_time'],
            base=aggr_metric_dict['PRM']['planning_time']
        )

    return aggr_metric_dict, delta_metric_dict


def run_test_case(case_map: dict, case_query: dict, case_rmp: RoadMap, case_o: RoundObstacle) -> dict:
    """
    Runs the test case and return recorded metrics. The following steps are operated:

    [1] Create a PRM solver with the given roadmap. Add the new obstacle to it.

    [2] Use the solver to run RRT and record metrics.

    [3] Use the solver to run PRM and record metrics.

    [4] Create a new solver with new obstacle. Use it to run R-PRM and record metrics.

    [5] Create a new solver with new obstacle. Use it to run RF-PRM and record metrics.

    [6] Create a new solver with new obstacle. Use it to run RFT-PRM and record metrics.
    :param case_map: map for the case
    :param case_query: query for the case
    :param case_rmp: roadmap for the case
    :param case_o: new obstacle for the case
    :return: the recorded metrics for the test case
    """
    # construct metrics dict for each algorithm
    metric_res = construct_metric_res(aggregate=False)

    # 1. create a PRM solver
    cur_prm = construct_prm_solver_with_obstacle(case_map=case_map, case_rmp=case_rmp, case_o=case_o)

    # 2. RRT
    rrt_path, rrt_cost, rrt_roadmap = cur_prm.plan_rrt(animation=False, **case_query)
    metric_res['RRT'] = record_path_res(metric_res['RRT'], path=rrt_path, cost=rrt_cost)
    metric_res['RRT']['planning_time'] = cur_prm.query_timer['rrt']

    # 3. PRM
    prm_path, prm_cost = cur_prm.plan(animation=False, repair=False, eval_freedom=False, **case_query)
    metric_res['PRM'] = record_path_res(metric_res['PRM'], path=prm_path, cost=prm_cost)
    metric_res['PRM']['planning_time'] = cur_prm.query_timer['shortest_path']

    # 4. R-PRM
    cur_prm = construct_prm_solver_with_obstacle(case_map=case_map, case_rmp=case_rmp, case_o=case_o)
    prm_path, prm_cost = cur_prm.plan(animation=False, repair=True, eval_freedom=False, **case_query)
    metric_res['R-PRM'] = record_path_res(metric_res['R-PRM'], path=prm_path, cost=prm_cost)
    metric_res['R-PRM']['planning_time'] = (cur_prm.query_timer['shortest_path'] + cur_prm.query_timer['repair'])
    metric_res['R-PRM']['extra'] = {
        'n_new_nodes': cur_prm.query_timer['n_new_nodes'],
        'repair_time': cur_prm.query_timer['repair']
    }

    # 5. RF-PRM
    cur_prm = construct_prm_solver_with_obstacle(case_map=case_map, case_rmp=case_rmp, case_o=case_o)
    prm_path, prm_cost = cur_prm.plan(animation=False, repair=True, eval_freedom=True, **case_query)
    metric_res['RF-PRM'] = record_path_res(metric_res['RF-PRM'], path=prm_path, cost=prm_cost)
    metric_res['RF-PRM']['planning_time'] = (cur_prm.query_timer['shortest_path'] + cur_prm.query_timer['repair'])
    metric_res['RF-PRM']['extra'] = {
        'n_new_nodes': cur_prm.query_timer['n_new_nodes'],
        'repair_time': cur_prm.query_timer['repair']
    }

    # TODO: 6. RFT-PRM

    # output
    return metric_res


def main(data_folder: str = DATA_DIR_DEFAULT, n_runs: int = N_RUNS_DEFAULT, logr: Optional[Logger] = None):
    if logr is None:
        logr = Logger(activate=False)

    cases_json = list_data_items(data_folder=data_folder)
    logr.log(f'> {len(cases_json)} cases collected.\n')

    logr.log('> Algorithms to evaluate:\n')
    logr.log('%s\n' % pformat(algorithm_dict, indent=4))

    logr.log('> Start running test cases.\n')
    overall_delta_res = construct_metric_res(aggregate=True)
    for cur_case_i in range(len(cases_json)):
        cur_case_json = cases_json[cur_case_i]
        log_horizontal_split(logr)
        logr.log(f'CASE {cur_case_i + 1}:\n')
        logr.log('%s\n' % pformat(cur_case_json, indent=4))

        aggr_res = construct_metric_res(aggregate=True)
        for run_i in range(n_runs):
            logr.log(f'CASE {cur_case_i + 1} - RUN {run_i + 1}:\n')
            cur_map, cur_query, cur_roadmap, cur_o = load_add_obstacle_case(test_case=cur_case_json)
            cur_res = run_test_case(case_map=cur_map, case_query=cur_query, case_rmp=cur_roadmap, case_o=cur_o)
            logr.log('%s\n' % pformat(cur_res, indent=4))
            aggr_res = aggregate_metric_res(aggr_metric_dict=aggr_res, cur_metric_res=cur_res)

        aggr_res, delta_res = postproc_aggr_metric_dict(aggr_metric_dict=aggr_res, n_runs=n_runs)
        logr.log(f'CASE {cur_case_i + 1} - OVERALL:\n')
        log_result(res=aggr_res, logr=logr)
        overall_delta_res = aggregate_metric_res(aggr_metric_dict=overall_delta_res, cur_metric_res=delta_res)

    overall_delta_res, _ = postproc_aggr_metric_dict(aggr_metric_dict=overall_delta_res, n_runs=len(cases_json),
                                                     handle_path_cost=False, calc_delta=False)
    log_horizontal_split(logr)
    logr.log(f'> Finish running {len(cases_json)} test cases. Final results:\n')
    log_result(res=overall_delta_res, logr=logr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-folder', type=str, default=DATA_DIR_DEFAULT,
                        help=f'where the generated problems are stored, default = {DATA_DIR_DEFAULT}')
    parser.add_argument('-n', '--n-runs', type=int, default=N_RUNS_DEFAULT,
                        help=f'how many repeated runs on each test case, default = {N_RUNS_DEFAULT}')
    parser.add_argument('-ld', '--log-dir', type=str, default=LOG_DIR_DEFAULT,
                        help=f'Specify where to create a log file if needed, default = {LOG_DIR_DEFAULT}')
    parser.add_argument('-tag', '--tag', type=str, default=TAG_DEFAULT,
                        help=f'Name tag for the model, default = {TAG_DEFAULT}')
    ARGS, _ = parser.parse_known_args()

    ARGS = vars(ARGS)

    logger = Logger(activate=True, logging_folder=ARGS['log_dir'], comment=ARGS['tag']) if ARGS['log_dir'] else Logger(activate=False)
    logger.log('> Main problem initialized.\n')
    logger.log(f'> Input variables: {ARGS}\n')

    main(data_folder=ARGS['data_folder'], n_runs=ARGS['n_runs'], logr=logger)

    logger.close()
