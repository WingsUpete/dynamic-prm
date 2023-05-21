from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from collections import OrderedDict
from enum import IntEnum

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
    def __init__(self):
        """
        Creates an Obstacle Dictionary that stores the obstacles in the environment.
        """
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

    def __len__(self):
        return len(self._o_dict)

    def __str__(self):
        return self._o_dict.__str__()
