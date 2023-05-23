import math
import time
from typing import Optional
import random

import matplotlib.pyplot as plt

from core.util.common import *
from core.util.graph import ObstacleDict, RoundObstacle
from core.util.plot import draw_path, draw_query_points, plot_circle
from .roadmap import RoadMapNode, RoadMap
from .shortest_path import dijkstra

__all__ = ['Prm']


class Prm:
    """
    PRM: Probabilistic Road Map

    Reference: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py#L195
    """
    def __init__(self,
                 map_range: list[float],
                 obstacles: ObstacleDict,
                 robot_radius: float,
                 init_n_samples: int = 300, init_n_neighbors: int = 10, max_edge_len: float = 30.0,
                 rrt_max_iter: int = 500, rrt_goal_sample_rate: float = 0.05,
                 rnd_seed: int = None):
        """
        Creates a PRM Solver for a robot to solve path-planning problems.
        :param map_range: the range of the map, as `[min, max]` for both `x` and `y`
        :param obstacles: ordered dict of round obstacles
        :param robot_radius: radius of the circle robot
        :param init_n_samples: initial number of points to sample
        :param init_n_neighbors: initial number of edges one sample point has
        :param max_edge_len: maximum edge length
        :param rrt_max_iter: maximum number of iterations for the RRT algorithm
        :param rrt_goal_sample_rate: specifies how frequent should RRT sample the goal point for analysis
        :param rnd_seed: random seed for sampler
        """
        # global timer: sec (unit)
        self.global_timer = {
            'sampling': 0.0,
            'edge_construction': 0.0,
            'collision_checking': {
                'n': 0,
                'total': 0.0,
                'max': float('-inf')
            },
        }
        self.query_timer: Optional[dict] = None

        self.map_min, self.map_max = map_range
        self.obstacles = obstacles

        self.robot_radius = robot_radius

        self.max_edge_len = max_edge_len

        self.rrt_max_iter = rrt_max_iter
        self.rrt_goal_sample_rate = rrt_goal_sample_rate

        self.rnd_seed = rnd_seed
        random.seed(self.rnd_seed)

        # build road map
        self.road_map: Optional[RoadMap] = None
        self._construct_road_map(n_samples=init_n_samples, n_neighbors=init_n_neighbors)

        # timer info
        self._postproc_timers()

    def plan(self, start: list[float], goal: list[float],
             repair: bool = False,
             animation: bool = True) -> (Optional[list[list[float]]], float):
        """
        Plans the route using PRM.

        :param start: starting position for the problem
        :param goal: goal position for the problem
        :param repair: specifies whether to repair PRM when new obstacles block the path that was feasible
        :param animation: enables animation or not
        :return: found feasible path as an ordered list of 2D points, or None if not found + path cost
        """
        if animation:
            self.draw_graph(start=start, goal=goal, road_map=self.road_map.get_clear_roadmap())

        self._reset_query_timer()
        try:
            if self._point_collides(x=start[0], y=start[1]) or self._point_collides(x=goal[0], y=goal[1]):
                # query points collide! No solution
                return None, -1

            start_sample_node = self._get_nearest_feasible_roadmap_node(point_x=start[0], point_y=start[1],
                                                                        from_point=True,
                                                                        road_map=self.road_map.get_clear_roadmap())
            if start_sample_node is None:
                return None, -1

            end_sample_node = self._get_nearest_feasible_roadmap_node(point_x=goal[0], point_y=goal[1],
                                                                      from_point=False,
                                                                      road_map=self.road_map.get_clear_roadmap())
            if end_sample_node is None:
                return None, -1

            t0 = time.time()
            path, cost = dijkstra(road_map=self.road_map.get_clear_roadmap(),
                                  start_uid=start_sample_node.node_uid, end_uid=end_sample_node.node_uid,
                                  animation=animation)
            self._record_time(timer=self.query_timer, metric='shortest_path', val=(time.time() - t0))

            # found path, normally output
            if path is not None:
                # Add paths and costs between the real start & end/goal point and the nearest feasible sample points
                path = [start] + [[cur_node.x, cur_node.y] for cur_node in path] + [goal]
                cost += cal_dist(from_x=start[0], from_y=start[1], to_x=start_sample_node.x, to_y=start_sample_node.y)
                cost += cal_dist(from_x=end_sample_node.x, from_y=end_sample_node.y, to_x=goal[0], to_y=goal[1])
                return path, cost

            # path not found
            if repair:
                if animation:
                    time.sleep(3)
                    self.draw_graph(start=start, goal=goal, road_map=self.road_map)

                t1 = time.time()
                # Rerun Dijkstra on the general road map to check if previously there was a path
                path, cost = dijkstra(road_map=self.road_map,
                                      start_uid=start_sample_node.node_uid, end_uid=end_sample_node.node_uid,
                                      animation=animation)
                if path is None:
                    # cannot find path even before nodes/edges are blocked, then nothing can be repaired.
                    return None, -1

                # otherwise: feasible path exists before
                # try to sample a starting point on the first path segment & an ending point on the last path segment
                # TODO
                return 'hi'


                self._record_time(timer=self.query_timer, metric='repair', val=(time.time() - t1))
        finally:
            self._postproc_timers()

    def plan_rrt(self, start: list[float], goal: list[float],
                 animation: bool = True, animate_interval: int = 5) -> (Optional[list[list[float]]],
                                                                        float,
                                                                        Optional[RoadMap]):
        """
        Plans the route using RRT.

        :param start: starting position for the problem
        :param goal: goal position for the problem
        :param animation: enables animation or not
        :param animate_interval: specifies how frequent (every x steps) should the graph be rendered
        :return: found feasible path as an ordered list of 2D points, or None if not found + path cost +
        explored road map
        """
        self._reset_query_timer()
        t0 = time.time()
        try:
            if self._point_collides(x=start[0], y=start[1]) or self._point_collides(x=goal[0], y=goal[1]):
                # query points collide! No solution
                return None, -1, None

            # create RRT Nodes for starting/goal points
            start_node = RoadMapNode(x=start[0], y=start[1])
            goal_node = RoadMapNode(x=goal[0], y=goal[1])

            return self._rrt_base(start=start_node, goal=goal_node,
                                  animation=animation, animate_interval=animate_interval)
        finally:
            self._record_time(timer=self.query_timer, metric='rrt', val=(time.time() - t0))
            self._postproc_timers()

    def add_obstacle_to_environment(self, obstacle: RoundObstacle) -> None:
        """
        Adds the obstacle to the environment and handle road map nodes/edges that are blocked. More details:

        1. Blocked nodes: distance to obstacle `d <= r + R`, where `r` is the robot radius and `R` is the obstacle
        radius.

        2. Extra blocked edges: all blocked nodes have their edges blocked. Moreover, need to check all nodes within
        distance `d <= r + R + L`, where `L` is the maximum edge length. For all of these nodes that are not blocked,
        check all of their edges: if they collide with the new obstacle, block this edge.
        :param obstacle: the obstacle to be added
        """
        # add the obstacle
        self.obstacles.add_obstacle(obstacle=obstacle)

        # mark all blocked points
        block_node_indices = self.road_map.find_points_within_r(point=[obstacle.x, obstacle.y],
                                                                r=(self.robot_radius + obstacle.r))
        for node_ind in block_node_indices:
            cur_node = self.road_map.get_node_by_index(index=node_ind)
            self.road_map.block_node(node_uid=cur_node.node_uid)

        # mark all blocked edges
        tmp_o_dict = ObstacleDict(map_range=[self.map_min, self.map_max], robot_radius=self.robot_radius)
        tmp_o_dict.add_obstacle(obstacle=obstacle)
        search_node_indices = self.road_map.find_points_within_r(point=[obstacle.x, obstacle.y],
                                                                 r=(self.robot_radius + obstacle.r + self.max_edge_len))
        for node_ind in search_node_indices:
            cur_node = self.road_map.get_node_by_index(index=node_ind)
            # blocked node -> skip
            if not cur_node.clear:
                continue
            # edge does not collide with new obstacle
            for to_uid in cur_node.to_node_uid_dict.keys():
                to_node = self.road_map.get()[to_uid]
                if not tmp_o_dict.reachable_without_collision(from_x=cur_node.x, from_y=cur_node.y,
                                                              to_x=to_node.x, to_y=to_node.y):
                    self.road_map.block_edge(from_uid=cur_node.node_uid, to_uid=to_uid)
            for from_uid in cur_node.from_node_uid_dict.keys():
                from_node = self.road_map.get()[from_uid]
                if not tmp_o_dict.reachable_without_collision(from_x=from_node.x, from_y=from_node.y,
                                                              to_x=cur_node.x, to_y=cur_node.y):
                    self.road_map.block_edge(from_uid=from_uid, to_uid=cur_node.node_uid)

    def _rrt_base(self, start: RoadMapNode, goal: RoadMapNode,
                  animation: bool = True, animate_interval: int = 5) -> (Optional[list[list[float]]],
                                                                         float,
                                                                         Optional[RoadMap]):
        """
        Plans the route using RRT. Compared to plan_rrt, the starting point and goal point are represented as road
        map nodes. This is used to record the uid of the two nodes for postprocessing.

        :param start: starting node
        :param goal: goal node
        :param animation: enables animation or not
        :param animate_interval: specifies how frequent (every x steps) should the graph be rendered
        :return: found feasible path as an ordered list of 2D points, or None if not found + path cost +
        explored road map
        """
        # create RRT Nodes for starting/goal points
        start_node = RoadMapNode(x=start.x, y=start.y, node_uid=start.node_uid)
        goal_node = RoadMapNode(x=goal.x, y=goal.y, node_uid=goal.node_uid)

        new_road_map = RoadMap(enable_kd_tree=False)
        new_road_map.add_node(node=start_node)
        for i in range(self.rrt_max_iter):
            # sample one node on the map
            rnd_node = self._get_rrt_rand_node(goal_node=goal_node)
            # get nearest neighbor of the sampled node
            nearest_node_ind = new_road_map.get_nearest_neighbor(point=[rnd_node.x, rnd_node.y])
            nearest_node = new_road_map.get_node_by_index(index=nearest_node_ind)

            # get the expected new node to reach
            new_node = self._steer_rrt(from_node=nearest_node, to_node=rnd_node)

            # check if `nearest_node` can reach `new_node` (collision check)
            if self.obstacles.reachable_without_collision(from_x=nearest_node.x, from_y=nearest_node.y,
                                                          to_x=new_node.x, to_y=new_node.y):
                # add new node and form edge
                new_road_map.add_node(node=new_node)
                new_road_map.add_edge(from_uid=nearest_node.node_uid, to_uid=new_node.node_uid)
                # goal check
                if new_node.euclidean_distance(goal_node) <= self.max_edge_len:
                    final_node = self._steer_rrt(from_node=new_node, to_node=goal_node)
                    if self.obstacles.reachable_without_collision(from_x=new_node.x, from_y=new_node.y,
                                                                  to_x=final_node.x, to_y=final_node.y):
                        # add final node and form edge
                        new_road_map.add_node(node=final_node)
                        new_road_map.add_edge(from_uid=new_node.node_uid, to_uid=final_node.node_uid)
                        # calculate final path and cost
                        path, cost = self._get_rrt_path_with_cost(road_map=new_road_map, final_node=final_node)
                        return path, cost, new_road_map

            # render graph
            if animation and i % animate_interval == 0:
                self.draw_graph(start=[start.x, start.y], goal=[goal.x, goal.y], road_map=new_road_map, pause=False)
                # render `rnd_node` and `new_node`
                plt.plot(rnd_node.x, rnd_node.y, '^c')  # ^c = cyan triangle
                if self.robot_radius > 0.0:
                    plot_circle(new_node.x, new_node.y, self.robot_radius, 'm', fill=False)  # m = magenta
                plt.pause(0.001)

        return None, -1, None

    def _get_rrt_rand_node(self, goal_node: RoadMapNode) -> RoadMapNode:
        """
        Generates an RRT Node.
        :param goal_node: goal node for this query
        :return:
        """
        if random.uniform(0, 1) <= self.rrt_goal_sample_rate:
            # sample goal point
            rnd = RoadMapNode(x=goal_node.x, y=goal_node.y)
        else:
            # sample random point on map
            rnd_c = self._sample_point()
            rnd = RoadMapNode(x=rnd_c[0], y=rnd_c[1])
        return rnd

    def _steer_rrt(self, from_node: RoadMapNode, to_node: RoadMapNode) -> RoadMapNode:
        """
        Steers the path from one node to the other, and retrieves a new node to replace the `to_node`. Essentially, it
        tries to reach from `from_node` and see how far it can reach along the direction to the `to_node`, given a
        maximum edge length.
        :param from_node: from this node
        :param to_node: to this node
        :return: a new node provided by the steering function to replace the `to_node`
        """
        d, theta = cal_dist_n_angle(from_x=from_node.x, from_y=from_node.y, to_x=to_node.x, to_y=to_node.y)
        extend_length = self.max_edge_len if self.max_edge_len <= d else d
        new_node = RoadMapNode(x=(from_node.x + extend_length * math.cos(theta)),
                               y=(from_node.y + extend_length * math.sin(theta)))
        return new_node

    @staticmethod
    def _get_rrt_path_with_cost(road_map: RoadMap, final_node: RoadMapNode) -> (list[list[float]], float):
        """
        Gets the path to the final node, along with its cost.
        :param road_map: give road map to operate on
        :param final_node: the final node of the path
        :return: the calculated path + cost
        """
        path = []
        cost = 0.0
        cur_node = final_node
        while True:
            path.append([cur_node.x, cur_node.y])
            if len(cur_node.from_node_uid_dict) == 0:
                # no parent
                break

            # Abnormal case coverage
            if len(cur_node.from_node_uid_dict) > 1:
                raise Exception('Why does your RRT path have node with more than 1 parents???')

            parent_uid = list(cur_node.from_node_uid_dict.keys())[0]
            parent_node = road_map.get()[parent_uid]
            cost += parent_node.euclidean_distance(other=cur_node)
            cur_node = parent_node

        path.reverse()
        return path, cost

    def _get_nearest_feasible_roadmap_node(self, point_x: float, point_y: float,
                                           from_point: bool, road_map: RoadMap) -> Optional[RoadMapNode]:
        """
        Finds the index of nearest feasible node on the road map from the given point. Feasibility indicates that the
        distance between two points is smaller than preset maximum, and going straight between two points do not
        collide with any obstacle.
        :param point_x: x coordinate of the given point
        :param point_y: y coordinate of the given point
        :param from_point: specifies the direction (if True, from this point to sample point; otherwise reverse)
        :param road_map: the road map to search from
        :return: the nearest feasible sample node, or None if not found
        """
        # sort distances from near to far
        _, indices = road_map.get_knn(point=[point_x, point_y], k=len(road_map))
        for cur_sample_id in indices:
            cur_sample_x = road_map.sample_x()[cur_sample_id]
            cur_sample_y = road_map.sample_y()[cur_sample_id]

            if from_point:
                from_x, from_y = point_x, point_y
                to_x, to_y = cur_sample_x, cur_sample_y
            else:
                from_x, from_y = cur_sample_x, cur_sample_y
                to_x, to_y = point_x, point_y
            d = cal_dist(from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y)
            if (d <= self.max_edge_len) and self._reachable_without_collision(from_x=from_x, from_y=from_y,
                                                                              to_x=to_x, to_y=to_y):
                # found nearest feasible sample point
                return road_map.get_node_by_index(index=cur_sample_id)

        # no feasible sample point
        return None

    def _construct_road_map(self, n_samples: int, n_neighbors: int):
        """
        Constructs the road map (nodes + edges).
        :param n_samples: initial number of points to sample
        :param n_neighbors: initial number of edges one sample point has
        """
        self._sample_points(n_samples=n_samples)
        self._construct_road_map_edges(n_neighbors=n_neighbors)

    def _sample_points(self, n_samples: int) -> None:
        """
        Samples n feasible points that do not collide with the obstacles, stores into a road map.
        :param n_samples: number of points to sample
        :return:
        """
        t0 = time.time()
        rmp = RoadMap()
        while len(rmp) != n_samples:
            tc = self._sample_point()
            tx, ty = tc[0], tc[1]

            if not self._point_collides(x=tx, y=ty):
                t_node = RoadMapNode(x=tx, y=ty)
                rmp.add_node(node=t_node)

        self._record_time(timer=self.global_timer, metric='sampling', val=(time.time() - t0))
        self.road_map = rmp

    def _sample_point(self) -> list[float]:
        """
        Uniformly samples a point on the map.
        :return: coordinates of the point as a list
        """
        return [
            random.uniform(self.map_min, self.map_max),
            random.uniform(self.map_min, self.map_max),
        ]

    def _construct_road_map_edges(self, n_neighbors: int) -> None:
        """
        Constructs edges for the road map using the sample points
        :param n_neighbors: initial number of edges one sample point has
        """
        t0 = time.time()
        for (ix, iy, i_uid) in zip(self.road_map.sample_x(), self.road_map.sample_y(), self.road_map.sample_uid()):
            # sort distances from near to far
            _, indices = self.road_map.get_knn(point=[ix, iy], k=len(self.road_map))
            n_new_edges_added = 0

            # starting from 1: ignore myself
            for ii in range(1, len(indices)):
                cur_n = self.road_map.get_node_by_index(index=indices[ii])
                nx = cur_n.x
                ny = cur_n.y
                n_uid = cur_n.node_uid

                # current examined node -> potential neighbor
                d = cal_dist(from_x=ix, from_y=iy, to_x=nx, to_y=ny)
                if (d <= self.max_edge_len) and self._reachable_without_collision(from_x=ix, from_y=iy, to_x=nx, to_y=ny):
                    self.road_map.add_edge(from_uid=i_uid, to_uid=n_uid)
                    n_new_edges_added += 1
                    if n_new_edges_added == n_neighbors:
                        break

        self._record_time(timer=self.global_timer, metric='edge_construction', val=(time.time() - t0))

    def draw_graph(self,
                   start: list[float] = None, goal: list[float] = None,
                   road_map: Optional[RoadMap] = None,
                   path: list[list[float]] = None,
                   pause: bool = True) -> None:
        """
        Draws the graph. More details:

        1. Obstacles: black points

        2. Map range: black lines

        3. Starting point: red triangle

        4. Goal point: red thin_diamond marker

        5. Sample points: cyan points

        6. Specified path: magenta solid line

        :param start: starting position of a query
        :param goal: goal position of a query
        :param road_map: the road map of the sample points
        :param path: specified path
        :param pause: specifies whether to pause and display the graph
        """
        plt.clf()

        # draw map edge and obstacles
        self.obstacles.draw_map_edge_n_obstacles()

        # draw road map
        if road_map is not None:
            road_map.draw_road_map()

        if path is not None:
            draw_path(path=path)

        # draw start & goal
        if (start is not None) and (goal is not None):
            draw_query_points(start=start, goal=goal)

        if pause:
            plt.pause(0.001)

    def _reachable_without_collision(self,
                                     from_x: float, from_y: float,
                                     to_x: float, to_y: float) -> bool:
        """
        Checks whether the robot will bump into an obstacle when she travels from one given point to the other given
        point. If collision check passes, return True!
        :param from_x: x coordinate of the `from_node`
        :param from_y: y coordinate of the `from_node`
        :param to_x: x coordinate of the `to_node`
        :param to_y: y coordinate of the `to_node`
        :return: collision check passes or not
        """
        return self.obstacles.reachable_without_collision(from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y,
                                                          collision_checking_func=self._point_collides)

    def _point_collides(self, x: float, y: float) -> bool:
        """
        Checks whether a point collides with any obstacle
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :return: whether collision happens, return True upon collision
        """
        t0 = time.time()
        try:
            return self.obstacles.point_collides(x=x, y=y)
        finally:
            delta_t = time.time() - t0
            self._record_time(timer=self.global_timer, metric='collision_checking', val=delta_t)

    def _reset_query_timer(self) -> None:
        """
        Resets the timer for a specific query
        :return:
        """
        self.query_timer = {
            'shortest_path': 0.0,
            'rrt': 0.0,
            'repair': 0.0
        }

    @staticmethod
    def _record_time(timer: dict, metric: str, val: float) -> None:
        """
        Records measured time in the timer
        :param timer: global timer or query timer
        :param metric: metric as key to record this value
        :param val: the value to be recorded
        """
        if timer is None:
            return
        if metric in {'collision_checking'}:
            timer[metric]['n'] += 1
            timer[metric]['total'] += val
            timer[metric]['max'] = max(timer[metric]['max'], val)
        else:
            timer[metric] = val

    def _postproc_timers(self) -> None:
        """
        Postprocesses timer info, like calculating average.
        """
        for timer in [self.global_timer, self.query_timer]:
            if timer is None:
                continue
            for metric in timer:
                if metric in {'collision_checking'}:
                    timer[metric]['avg'] = timer[metric]['total'] / timer[metric]['n']
