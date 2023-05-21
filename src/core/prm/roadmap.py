from collections import OrderedDict
from typing import Optional

import numpy as np
from scipy.spatial import KDTree

from core.util import Node2D

__all__ = ['RoadMapNode', 'RoadMap']


class RoadMapNode(Node2D):
    """ Node for Road Map """
    def __init__(self, x: float, y: float, node_uid: str = None):
        """
        Creates a Road Map Node.
        :param x: x coordinate of the node
        :param y: y coordinate of the node
        :param node_uid: uid of the node
        """
        super().__init__(x=x, y=y, node_uid=node_uid)
        if not self.node_uid:
            self.node_uid = self.node_id

        # storing neighbors
        self.from_node_uid_set: set[str] = set()
        self.to_node_uid_set: set[str] = set()


class RoadMap:
    """ Road Map """
    def __init__(self):
        """
        Creates a Road Map that stores the sample point graph as well as a KD Tree for sample points.
        """
        # node_uid -> node
        self._road_map: OrderedDict[str, RoadMapNode] = OrderedDict()

        # ordered list of separated sample coordinates transformed from the road map
        self._sample_x: list[float] = []
        self._sample_y: list[float] = []

        # ordered list of sample node uid transformed from the road map
        self._sample_uid: list[str] = []

        # kd tree for sample points
        self._kd_tree: Optional[KDTree] = None

        # if road map modified, need to update other variables
        self._modified = True

    def get(self):
        """
        Gets the road map.
        :return: road map
        """
        return self._road_map

    def get_node_by_index(self, index: int) -> RoadMapNode:
        """
        Retrieves the node according to the node index on the ordered list transformed from the road map
        :param index: node index on the ordered list
        :return: the corresponding node
        """
        return self.get()[self.sample_uid()[index]]

    def get_knn(self, point: list[float], k: int) -> (list[float], list[int]):
        dists, indices = self.kd_tree().query(point, k=k)
        dists = [dists] if type(dists) == float else dists
        indices = [indices] if type(indices) == int else indices
        return dists, indices

    def add_node(self, node: RoadMapNode) -> None:
        """
        Adds a node to the road map.
        :param node: given node to be added
        """
        self.get()[node.node_uid] = node
        self._modified = True

    def remove_node(self, node_uid: str) -> None:
        """
        Removes a node from the road map
        :param node_uid: uid of the node to be removed
        """
        del self.get()[node_uid]
        self._modified = True

    def add_edge(self, from_uid: str, to_uid: str) -> None:
        """
        Adds an edge from one node to the other.
        :param from_uid: uid of the source node
        :param to_uid: uid of the destination node
        """
        self.get()[from_uid].to_node_uid_set.add(to_uid)
        self.get()[to_uid].from_node_uid_set.add(from_uid)
        self._modified = True

    def remove_edge(self, from_uid: str, to_uid: str) -> None:
        """
        Removes an edge from one node to the other.
        :param from_uid: uid of the source node
        :param to_uid: uid of the destination node
        """
        self.get()[from_uid].to_node_uid_set.remove(to_uid)
        self.get()[to_uid].from_node_uid_set.remove(from_uid)
        self._modified = True

    def _update_coordinate_lists(self) -> None:
        """
        Updates the coordinate lists `sample_x` and `sample_y`.
        """
        self._sample_x = [node.x for node in self.get().values()]
        self._sample_y = [node.y for node in self.get().values()]
        self._sample_uid = [node.node_uid for node in self.get().values()]
        self._kd_tree = KDTree(np.vstack((self._sample_x, self._sample_y)).T)
        self._modified = False

    def sample_x(self) -> list[float]:
        """
        Gets x coordinate ordered list of the sample points.
        :return: x coordinate ordered list of the sample points
        """
        if self._modified:
            self._update_coordinate_lists()
        return self._sample_x

    def sample_y(self) -> list[float]:
        """
        Gets y coordinate ordered list of the sample points.
        :return: y coordinate ordered list of the sample points
        """
        if self._modified:
            self._update_coordinate_lists()
        return self._sample_y

    def sample_uid(self) -> list[str]:
        """
        Gets node uid ordered list of the sample points.
        :return: node uid ordered list of the sample points
        """
        if self._modified:
            self._update_coordinate_lists()
        return self._sample_uid

    def kd_tree(self) -> KDTree:
        """
        Gets KD Tree for the sample points.
        :return: sample KD Tree
        """
        if self._modified:
            self._update_coordinate_lists()
        return self._kd_tree

    def __len__(self):
        return len(self._road_map)

    def __str__(self):
        return self._road_map.__str__()
