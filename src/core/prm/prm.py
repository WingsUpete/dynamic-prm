import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from .shortest_path import dijkstra
from core.util import plot_circle

__all__ = ['Prm']


class Prm:
    """
    PRM: Probabilistic Road Map

    Reference: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py#L195
    """
    def __init__(self,
                 map_range: list[float],
                 obstacle_xs: list[float], obstacle_ys: list[float], obstacle_rs: list[float],
                 robot_radius: float,
                 n_samples: int = 500, n_neighbors: int = 10, max_edge_len: float = 30.0,
                 rnd_seed: int = None, rng=None):
        """
        Creates a PRM Solver for a robot to solve path-planning problems.
        :param map_range: the range of the map, as `[min, max]` for both `x` and `y`
        :param obstacle_xs: list of x coordinates of the round obstacles
        :param obstacle_ys: list of y coordinates of the round obstacles
        :param obstacle_rs: list of r as radii of the round obstacles
        :param robot_radius: radius of the circle robot
        :param n_samples: number of points to sample
        :param n_neighbors: number of edges one sample point has
        :param max_edge_len: maximum edge length
        :param rnd_seed: random seed for sampler
        :param rng: specifies a random generator to use
        """
        self.map_min, self.map_max = map_range
        self.obstacle_x_list = obstacle_xs
        self.obstacle_y_list = obstacle_ys
        self.obstacle_r_list = obstacle_rs

        self.robot_radius = robot_radius

        self.n_samples = n_samples
        self.n_neighbors = n_neighbors
        self.max_edge_len = max_edge_len

        self.rnd_seed = rnd_seed
        np.random.seed(self.rnd_seed)
        self.rng = rng

        # # kd-tree for obstacles
        # self.obstacle_kd_tree = KDTree(np.vstack((self.obstacle_x_list, self.obstacle_y_list)).T)

        # sample n points
        self.sample_x, self.sample_y = self.sample_points()
        self.sample_kd_tree = KDTree(np.vstack((self.sample_x, self.sample_y)).T)

        # Generate road map
        self.road_map = self.generate_road_map()

    def plan(self, start: list[float], goal: list[float],
             animation: bool = True) -> (Optional[list[list[float]]], float):
        """
        Plans the route using PRM.

        :param start: starting position for the problem
        :param goal: goal position for the problem
        :param animation: enables animation or not
        :return: found feasible path as an ordered list of 2D points, or None if not found + path cost
        """
        if animation:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None]
            )
            self.draw_graph(start=start, goal=goal,
                            sample_point_x_list=self.sample_x, sample_point_y_list=self.sample_y,
                            road_map=self.road_map)

        if self.point_collides(x=start[0], y=start[1]) or self.point_collides(x=goal[0], y=goal[1]):
            # query points collide! No solution
            return None, -1

        start_sample_id = self.get_nearest_feasible_sample_point_id(point_x=start[0], point_y=start[1], from_point=True)
        if start_sample_id == -1:
            return None, -1

        end_sample_id = self.get_nearest_feasible_sample_point_id(point_x=goal[0], point_y=goal[1], from_point=False)
        if end_sample_id == -1:
            return None, -1

        path, cost = dijkstra(sample_x=self.sample_x, sample_y=self.sample_y, road_map=self.road_map,
                              start_id=start_sample_id, end_id=end_sample_id, animation=animation)

        if path is None:
            return None, -1

        # Add paths and costs between the real start & end/goal point and the nearest feasible sample points
        path = [start] + path + [goal]
        cost += self.cal_dist_n_angle(from_x=start[0], from_y=start[1],
                                      to_x=self.sample_x[start_sample_id], to_y=self.sample_y[start_sample_id])[0]
        cost += self.cal_dist_n_angle(from_x=self.sample_x[end_sample_id], from_y=self.sample_y[end_sample_id],
                                      to_x=goal[0], to_y=goal[1])[0]

        return path, cost

    def get_nearest_feasible_sample_point_id(self, point_x: float, point_y: float, from_point: bool) -> int:
        """
        Finds the index of nearest feasible sample point from the given point. Feasibility indicates that
        :param point_x: x coordinate of the given point
        :param point_y: y coordinate of the given point
        :param from_point: specifies the direction (if True, from this point to sample point; otherwise reverse)
        :return: id of the nearest feasible sample point, or -1 if not found
        """
        # sort distances from near to far
        dists, indices = self.sample_kd_tree.query([point_x, point_y], k=self.n_samples)
        for cur_sample_id in indices:
            cur_sample_x = self.sample_x[cur_sample_id]
            cur_sample_y = self.sample_y[cur_sample_id]

            if from_point:
                from_x, from_y = point_x, point_y
                to_x, to_y = cur_sample_x, cur_sample_y
            else:
                from_x, from_y = cur_sample_x, cur_sample_y
                to_x, to_y = point_x, point_y
            d, _ = self.cal_dist_n_angle(from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y)
            if (d <= self.max_edge_len) and self.pass_collision_check(from_x=from_x, from_y=from_y,
                                                                      to_x=to_x, to_y=to_y):
                # found nearest feasible sample point
                return cur_sample_id

        # no feasible sample point
        return -1

    def sample_points(self) -> (list[float], list[float]):
        """
        Samples n feasible points that do not collide with the obstacles
        :return: x coordinate list & y coordinate list of n sampled points
        """
        sample_x, sample_y = [], []
        rng = np.random.default_rng() if self.rng is None else self.rng
        while len(sample_x) < self.n_samples:
            tx = self.map_min + (rng.random() * (self.map_max - self.map_min))
            ty = self.map_min + (rng.random() * (self.map_max - self.map_min))

            if not self.point_collides(x=tx, y=ty):
                sample_x.append(tx)
                sample_y.append(ty)

            # dist, _ = self.obstacle_kd_tree.query([tx, ty])
            # if dist > self.robot_radius:
            #     # will not collide with the nearest obstacle -> feasible sample point
            #     sample_x.append(tx)
            #     sample_y.append(ty)

        return sample_x, sample_y

    def draw_graph(self,
                   start: list[float] = None, goal: list[float] = None,
                   sample_point_x_list: list[float] = None, sample_point_y_list: list[float] = None,
                   road_map: list[list[int]] = None,
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
        :param sample_point_x_list: x coordinate list of the sample points
        :param sample_point_y_list: y coordinate list of the sample points
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
        for oi in range(len(self.obstacle_x_list)):
            if self.obstacle_r_list[oi] > 0.0:
                plot_circle(x=self.obstacle_x_list[oi], y=self.obstacle_y_list[oi], r=self.obstacle_r_list[oi],
                            c=obstacle_color, fill=True)
            else:
                plt.plot([self.obstacle_x_list[oi]], [self.obstacle_y_list[oi]], f'.{obstacle_color}')  # . = point

        # draw road map
        if road_map is not None:
            for node_i in range(len(road_map)):
                for node_j in road_map[node_i]:
                    plt.plot([self.sample_x[node_i], self.sample_x[node_j]],
                             [self.sample_y[node_i], self.sample_y[node_j]], '-y', alpha=0.2)  # -k = yellow solid line

        # draw sample points
        if (sample_point_x_list is not None) and (sample_point_y_list is not None):
            plt.plot(sample_point_x_list, sample_point_y_list, '.c')    # .c = cyan points

        if path is not None:
            self.draw_path(path=path)

        # draw start & goal
        if (start is not None) and (goal is not None):
            plt.plot(start[0], start[1], '^r')  # ^r = red triangle
            plt.plot(goal[0], goal[1], 'dr')    # dr = red thin_diamond marker

        plt.pause(0.001)

    @staticmethod
    def draw_path(path: list[list[float]]) -> None:
        """
        Draws the path as magenta solid line.
        :param path: specified path
        """
        plt.plot([x for (x, _) in path], [y for (_, y) in path], '-m')  # -m = magenta solid line
        plt.pause(0.001)

    @staticmethod
    def cal_dist_n_angle(from_x: float, from_y: float,
                         to_x: float, to_y: float) -> (float, float):
        """
        Calculates the distance and angle from one node to the other.
        :param from_x: x coordinate of the `from_node`
        :param from_y: y coordinate of the `from_node`
        :param to_x: x coordinate of the `to_node`
        :param to_y: y coordinate of the `to_node`
        :return: `d` as distance, `theta` as angle
        """
        dx = to_x - from_x
        dy = to_y - from_y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def generate_road_map(self) -> list[list[int]]:
        """
        Generates the road map using the sample points
        :return: generated road map, with each item storing the edges it connects to other nodes
        """
        road_map = []
        for (i, ix, iy) in zip(range(self.n_samples), self.sample_x, self.sample_y):
            # sort distances from near to far
            dists, indices = self.sample_kd_tree.query([ix, iy], k=self.n_samples)
            edge_id_list = []

            # starting from 1: ignore myself
            for ii in range(1, len(indices)):
                nx = self.sample_x[indices[ii]]
                ny = self.sample_y[indices[ii]]

                d, _ = self.cal_dist_n_angle(from_x=ix, from_y=iy, to_x=nx, to_y=ny)
                if (d <= self.max_edge_len) and self.pass_collision_check(from_x=ix, from_y=iy, to_x=nx, to_y=ny):
                    edge_id_list.append(indices[ii])

                if len(edge_id_list) >= self.n_neighbors:
                    break

            road_map.append(edge_id_list)

        return road_map

    def pass_collision_check(self,
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
        d, theta = self.cal_dist_n_angle(from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y)
        path_resolution = self.robot_radius
        n_steps = round(d / path_resolution)

        cur_x = from_x
        cur_y = from_y
        for _ in range(n_steps):
            if self.point_collides(x=cur_x, y=cur_y):
                return False

            # dist, _ = self.obstacle_kd_tree.query([cur_x, cur_y])
            # if dist <= self.robot_radius:
            #     # collide
            #     return False

            cur_x += path_resolution * math.cos(theta)
            cur_y += path_resolution * math.sin(theta)

        if (cur_x != to_x) or (cur_y != to_y):
            # `!(cur_x == to_x and cur_y == to_y)`
            # currently not reaching `to_node`, should also check `to_node` (TODO: maybe not since it is a sample point)
            if self.point_collides(x=to_x, y=to_y):
                return False

            # dist, _ = self.obstacle_kd_tree.query([to_x, to_y])
            # if dist <= self.robot_radius:
            #     # collide
            #     return False

        return True

    def point_collides(self, x: float, y: float) -> bool:
        """
        Checks whether a point collides with any obstacle
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :return: whether collision happens, return True upon collision
        """
        for oi in range(len(self.obstacle_x_list)):
            cur_ox, cur_oy, cur_or = self.obstacle_x_list[oi], self.obstacle_y_list[oi], self.obstacle_r_list[oi]
            cur_d, _ = self.cal_dist_n_angle(from_x=x, from_y=y, to_x=cur_ox, to_y=cur_oy)
            if cur_d <= self.robot_radius + cur_or:
                # collision!
                return True

        return False
