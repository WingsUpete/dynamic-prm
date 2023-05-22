from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from collections import OrderedDict
from enum import IntEnum
import math
from typing import Callable

import matplotlib.pyplot as plt

from core.util.common import *
from core.util.plot import plot_circle

__all__ = ['Node2D', 'RoundObstacle', 'ObstacleDict', 'ObstacleType']


class Node2D:
    def __init__(self, x: float, y: float, node_id: str = None, node_uid: str = None):
        """
        Creates a 2D Node on the map.

        NOTE: when there are two nodes `n1` and `n2`, `n1 == n2` is true as long as their coordinates are identical,
        but that does not necessarily mean that their `node_id`s match.
        :param x: `x` coordinate
        :param y: `y` coordinate
        :param node_id: node ID as a SHA256 string to uniquely identify the node
        :param node_uid: user-provided node ID as a string
        """
        self.x = x
        self.y = y

        self.node_id = self._generate_id() if node_id is None else node_id

        self.node_uid = node_uid

    def _generate_id(self) -> str:
        """
        Generates an identification string using SHA256. The semantic string for hashing includes the coordinates,
        as well as the current timestamp.
        :return: sha256 hash string as id
        """
        semantic_str = f'{self.x}{self.y}' + datetime.now().strftime('%Y%m%d %H:%M:%S')
        hash_str = sha256(semantic_str.encode()).hexdigest()
        return hash_str

    def euclidean_squared_distance(self, other: Node2D) -> float:
        return (self.x - other.x)**2 + (self.y - other.y)**2

    def euclidean_distance(self, other: Node2D) -> float:
        return self.euclidean_squared_distance(other=other)**0.5

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.node_id}: ({self.x}, {self.y})>'


class ObstacleType(IntEnum):
    NORMAL = 0
    MAP_EDGE = 1


class RoundObstacle(Node2D):
    """ Round Obstacle """
    def __init__(self, x: float, y: float, r: float, obstacle_type: int, obstacle_uid: str = None):
        """
        Creates a round obstacle.
        :param x: x coordinate of the obstacle
        :param y: y coordinate of the obstacle
        :param r: radius of the obstacle
        :param obstacle_type: type of the obstacle (refer to `ObstacleType`)
        :param obstacle_uid: uid of the obstacle
        """
        super().__init__(x=x, y=y, node_uid=obstacle_uid)
        if not self.node_uid:
            self.node_uid = self.node_id

        self.obstacle_uid = self.node_uid
        self.obstacle_type = obstacle_type
        self.r = r

    def update_uid(self, uid: str) -> None:
        """
        Updates `node_uid` and `obstacle_uid` at the same time.
        :param uid: given uid to update
        """
        self.node_uid = uid
        self.obstacle_uid = uid


class ObstacleDict:
    """ Obstacle Dictionary """
    def __init__(self, map_range: list[float], robot_radius: float):
        """
        Creates an Obstacle Dictionary that stores the obstacles in the environment.
        :param map_range: the range of the map, as `[min, max]` for both `x` and `y`
        :param robot_radius: radius of the circle robot
        """
        self.map_min, self.map_max = map_range
        self.robot_radius = robot_radius

        # obstacle_uid -> obstacle
        self._o_dict: OrderedDict[str, RoundObstacle] = OrderedDict()

        # ordered list of separated obstacle coordinates + radii transformed from the road map
        self._o_x: list[float] = []
        self._o_y: list[float] = []
        self._o_r: list[float] = []

        # ordered list of sample node uid transformed from the road map
        self._o_uid: list[str] = []

        # if road map modified, need to update other variables
        self._modified = True

        # Generates map edge point obstacles to restrict the robot within the map
        self.n_map_edge_obstacles = self._gen_map_edge_obstacles()

    def get(self):
        """
        Gets the road map.
        :return: road map
        """
        return self._o_dict

    def get_node_by_index(self, index: int) -> RoundObstacle:
        """
        Retrieves the node according to the node index on the ordered list transformed from the road map
        :param index: node index on the ordered list
        :return: the corresponding node
        """
        return self.get()[self.o_uid()[index]]

    def get_n_normal_obstacles(self) -> int:
        """
        Retrieves the number of normal obstacles (that are not map edge point obstacles)
        :return: the number as int
        """
        return len(self._o_dict) - self.n_map_edge_obstacles

    def add_obstacle(self, obstacle: RoundObstacle) -> None:
        """
        Adds an obstacle to the map.
        :param obstacle: given obstacle to be added
        """
        self.get()[obstacle.obstacle_uid] = obstacle
        self._modified = True

    def remove_obstacle(self, obstacle_uid: str) -> None:
        """
        Removes an obstacle from the map
        :param obstacle_uid: uid of the obstacle to be removed
        """
        del self.get()[obstacle_uid]
        self._modified = True

    def _gen_map_edge_obstacles(self, shrink_factor: float = 0.9) -> int:
        """
        Generates point obstacles at the edge of the map, ensuring that the robot cannot leave the map. Adds these
        obstacles to the dict.

        Suppose the obstacle interval (distance between two consecutive obstacles) is `d`, then it requires that
        `d < 2r`. Using a shrink factor `0 << t < 1`, we can write `d <= 2tr`. Now suppose the map range is of `l`
        length, then the number of obstacles we have to place is `n = l/d + 1 >= l/2tr + 1`, which leads to
        `n = ceil(l/2tr) + 1`.
        :param shrink_factor: how much should the obstacle interval instance be shrunk (`t`) to avoid the robot through
        :return: number of map edge obstacles: `4n - 4` for `n` obstacles on each edge
        """
        map_range_len = self.map_max - self.map_min  # l
        n_gaps = math.ceil(map_range_len / (2 * shrink_factor * self.robot_radius))
        d_real = map_range_len / n_gaps
        n_obstacles = n_gaps + 1

        gen_ref_dict = {
            'bottom': {
                'range': [0, n_obstacles],
                'fx': lambda oi: self.map_min + d_real * oi,
                'fy': lambda _: self.map_min
            },
            'left': {
                # bottom one already added, so skip it
                'range': [1, n_obstacles],
                'fx': lambda _: self.map_min,
                'fy': lambda oi: self.map_min + d_real * oi
            },
            'right': {
                # bottom one already added, so skip it
                'range': [1, n_obstacles],
                'fx': lambda _: self.map_max,
                'fy': lambda oi: self.map_min + d_real * oi
            },
            'top': {
                # left & right ones already added, so skip both of them
                'range': [1, n_obstacles - 1],
                'fx': lambda oi: self.map_min + d_real * oi,
                'fy': lambda _: self.map_max
            }
        }

        for cur_config in gen_ref_dict.values():
            for i in range(*cur_config['range']):
                cur_o = RoundObstacle(x=cur_config['fx'](i), y=cur_config['fy'](i), r=0,
                                      obstacle_type=ObstacleType.MAP_EDGE)
                self.add_obstacle(obstacle=cur_o)

        return 4 * n_obstacles - 4

    def _update_dependent_vars(self) -> None:
        """
        Updates the variables dependent on the obstacle dictionary.
        """
        self._o_x = [obstacle.x for obstacle in self.get().values()]
        self._o_y = [obstacle.y for obstacle in self.get().values()]
        self._o_r = [obstacle.r for obstacle in self.get().values()]
        self._o_uid = [obstacle.obstacle_uid for obstacle in self.get().values()]
        self._modified = False

    def o_x(self) -> list[float]:
        """
        Gets x coordinate ordered list of the obstacles.
        :return: x coordinate ordered list of the obstacles
        """
        if self._modified:
            self._update_dependent_vars()
        return self._o_x

    def o_y(self) -> list[float]:
        """
        Gets y coordinate ordered list of the obstacles.
        :return: y coordinate ordered list of the obstacles
        """
        if self._modified:
            self._update_dependent_vars()
        return self._o_y

    def o_r(self) -> list[float]:
        """
        Gets radius ordered list of the obstacles.
        :return: radius ordered list of the obstacles
        """
        if self._modified:
            self._update_dependent_vars()
        return self._o_r

    def o_uid(self) -> list[str]:
        """
        Gets obstacle uid ordered list of the obstacles.
        :return: obstacle uid ordered list of the obstacles
        """
        if self._modified:
            self._update_dependent_vars()
        return self._o_uid

    def point_collides(self, x: float, y: float) -> bool:
        """
        Checks whether a point collides with any obstacle.
        :param x: x coordinate of the point
        :param y: y coordinate of the point
        :return: whether collision happens, return True upon collision
        """
        for (ox, oy, o_radius) in zip(self.o_x(), self.o_y(), self.o_r()):
            cur_d = cal_dist(from_x=x, from_y=y, to_x=ox, to_y=oy)
            if cur_d <= self.robot_radius + o_radius:
                # collision!
                return True

        return False

    def reachable_without_collision(self,
                                    from_x: float, from_y: float,
                                    to_x: float, to_y: float,
                                    collision_checking_func: Callable = point_collides) -> bool:
        """
        Checks whether the robot will bump into an obstacle when she travels from one given point to the other given
        point. If collision check passes, return True!
        :param from_x: x coordinate of the `from_node`
        :param from_y: y coordinate of the `from_node`
        :param to_x: x coordinate of the `to_node`
        :param to_y: y coordinate of the `to_node`
        :param collision_checking_func: specifies a collision_checking_function (default to my own)
        :return: collision check passes or not
        """
        d, theta = cal_dist_n_angle(from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y)
        path_resolution = self.robot_radius
        n_steps = round(d / path_resolution)

        cur_x = from_x
        cur_y = from_y
        for _ in range(n_steps):
            if collision_checking_func(x=cur_x, y=cur_y):
                return False

            cur_x += path_resolution * math.cos(theta)
            cur_y += path_resolution * math.sin(theta)

        if (cur_x != to_x) or (cur_y != to_y):
            # `!(cur_x == to_x and cur_y == to_y)`
            # currently not reaching `to_node`, should also check `to_node`
            if collision_checking_func(x=to_x, y=to_y):
                return False

        return True

    def draw_map_edge_n_obstacles(self, c: str = 'k', padding: float = 3) -> None:
        """
        Draws the map edge and obstacles with the specified color.
        :param c: specified color
        :param padding: padding of the plot around the map
        """
        plt.clf()

        # set global map info
        plt.axis('equal')
        plt.xlim([self.map_min - padding, self.map_max + padding])
        plt.ylim([self.map_min - padding, self.map_max + padding])
        plt.grid(False)

        # draw map edge: starting from left bottom, go counter-clockwise
        map_vx_list = [self.map_min, self.map_max, self.map_max, self.map_min]
        map_vy_list = [self.map_min, self.map_min, self.map_max, self.map_max]
        map_vx_list.append(map_vx_list[0])
        map_vy_list.append(map_vy_list[0])
        plt.plot(map_vx_list, map_vy_list, f'-{c}')    # - = solid line

        # draw obstacles
        for (ox, oy, o_radius) in zip(self.o_x(), self.o_y(), self.o_r()):
            if o_radius > 0.0:
                plot_circle(x=ox, y=oy, r=o_radius, c=c, fill=True)
            else:
                plt.plot([ox], [oy], f'.{c}')  # . = point

        plt.pause(0.001)

    def __len__(self):
        return len(self._o_dict)

    def __str__(self):
        return self._o_dict.__str__()

    def __repr__(self):
        return self._o_dict.__repr__()
