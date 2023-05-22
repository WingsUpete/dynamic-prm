import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from core.util.common import cal_dist
from core.util.graph import ObstacleDict
from core.util.plot import plot_circle, draw_path
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
                 init_n_samples: int = 500, init_n_neighbors: int = 10, max_edge_len: float = 30.0,
                 rnd_seed: int = None, rng=None):
        """
        Creates a PRM Solver for a robot to solve path-planning problems.
        :param map_range: the range of the map, as `[min, max]` for both `x` and `y`
        :param obstacles: ordered dict of round obstacles
        :param robot_radius: radius of the circle robot
        :param init_n_samples: initial number of points to sample
        :param init_n_neighbors: initial number of edges one sample point has
        :param max_edge_len: maximum edge length
        :param rnd_seed: random seed for sampler
        :param rng: specifies a random generator to use
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

        self.rnd_seed = rnd_seed
        np.random.seed(self.rnd_seed)
        self.rng = rng

        # build road map
        self.road_map: Optional[RoadMap] = None
        self._construct_road_map(n_samples=init_n_samples, n_neighbors=init_n_neighbors)

        # timer info
        self._postproc_timers()

    def plan(self, start: list[float], goal: list[float],
             animation: bool = True) -> (Optional[list[list[float]]], float):
        """
        Plans the route using PRM.

        :param start: starting position for the problem
        :param goal: goal position for the problem
        :param animation: enables animation or not
        :return: found feasible path as an ordered list of 2D points, or None if not found + path cost
        """
        self._reset_query_timer()
        try:
            if animation:
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None]
                )
                self.draw_graph(start=start, goal=goal, road_map=self.road_map)

            if self._point_collides(x=start[0], y=start[1]) or self._point_collides(x=goal[0], y=goal[1]):
                # query points collide! No solution
                return None, -1

            start_sample_node = self.get_nearest_feasible_sample_node(point_x=start[0], point_y=start[1], from_point=True)
            if start_sample_node is None:
                return None, -1

            end_sample_node = self.get_nearest_feasible_sample_node(point_x=goal[0], point_y=goal[1], from_point=False)
            if end_sample_node is None:
                return None, -1

            t0 = time.time()
            path, cost = dijkstra(road_map=self.road_map,
                                  start_uid=start_sample_node.node_uid, end_uid=end_sample_node.node_uid,
                                  animation=animation)
            self._record_time(timer=self.query_timer, metric='shortest_path', val=(time.time() - t0))

            if path is None:
                return None, -1

            # Add paths and costs between the real start & end/goal point and the nearest feasible sample points
            path = [start] + path + [goal]
            cost += cal_dist(from_x=start[0], from_y=start[1], to_x=start_sample_node.x, to_y=start_sample_node.y)
            cost += cal_dist(from_x=end_sample_node.x, from_y=end_sample_node.y, to_x=goal[0], to_y=goal[1])

            return path, cost
        finally:
            self._postproc_timers()

    def get_nearest_feasible_sample_node(self, point_x: float, point_y: float,
                                         from_point: bool) -> Optional[RoadMapNode]:
        """
        Finds the index of nearest feasible sample point from the given point. Feasibility indicates that
        :param point_x: x coordinate of the given point
        :param point_y: y coordinate of the given point
        :param from_point: specifies the direction (if True, from this point to sample point; otherwise reverse)
        :return: the nearest feasible sample node, or None if not found
        """
        # sort distances from near to far
        _, indices = self.road_map.get_knn(point=[point_x, point_y], k=len(self.road_map))
        for cur_sample_id in indices:
            cur_sample_x = self.road_map.sample_x()[cur_sample_id]
            cur_sample_y = self.road_map.sample_y()[cur_sample_id]

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
                return self.road_map.get_node_by_index(index=cur_sample_id)

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
        rng = np.random.default_rng() if self.rng is None else self.rng
        while len(rmp) != n_samples:
            tx = self.map_min + (rng.random() * (self.map_max - self.map_min))
            ty = self.map_min + (rng.random() * (self.map_max - self.map_min))

            if not self._point_collides(x=tx, y=ty):
                t_node = RoadMapNode(x=tx, y=ty)
                rmp.add_node(node=t_node)

        self._record_time(timer=self.global_timer, metric='sampling', val=(time.time() - t0))
        self.road_map = rmp

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
                   padding: float = 3) -> None:
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
        :param padding: padding of the plot around the map
        """
        plt.clf()

        # set global map info
        plt.axis('equal')
        plt.xlim([self.map_min - padding, self.map_max + padding])
        plt.ylim([self.map_min - padding, self.map_max + padding])
        plt.grid(False)

        # draw map range: starting from left bottom, go counter-clockwise
        map_vx_list = [self.map_min, self.map_max, self.map_max, self.map_min]
        map_vy_list = [self.map_min, self.map_min, self.map_max, self.map_max]
        map_vx_list.append(map_vx_list[0])
        map_vy_list.append(map_vy_list[0])
        plt.plot(map_vx_list, map_vy_list, '-k')    # -k = black solid line

        # draw obstacles
        obstacle_color = 'k'   # k = black
        for (ox, oy, o_radius) in zip(self.obstacles.o_x(), self.obstacles.o_y(), self.obstacles.o_r()):
            if o_radius > 0.0:
                plot_circle(x=ox, y=oy, r=o_radius, c=obstacle_color, fill=True)
            else:
                plt.plot([ox], [oy], f'.{obstacle_color}')  # . = point

        # draw road map
        if road_map is not None:
            # edges
            for (ix, iy, i_uid) in zip(road_map.sample_x(), road_map.sample_y(), road_map.sample_uid()):
                for j_uid in road_map.get()[i_uid].to_node_uid_set:
                    j_node = road_map.get()[j_uid]
                    plt.plot([ix, j_node.x],
                             [iy, j_node.y], '-y', alpha=0.2)  # -k = yellow solid line
            # nodes
            plt.plot(road_map.sample_x(), road_map.sample_y(), '.c')  # .c = cyan points

        if path is not None:
            draw_path(path=path)

        # draw start & goal
        if (start is not None) and (goal is not None):
            plt.plot(start[0], start[1], '^r')  # ^r = red triangle
            plt.plot(goal[0], goal[1], 'dr')    # dr = red thin_diamond marker

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
