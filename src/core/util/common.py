from __future__ import annotations

from datetime import datetime
from hashlib import sha256

__all__ = ['Node2D']


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
