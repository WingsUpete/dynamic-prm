from collections import OrderedDict
from typing import Optional

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

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

    def add_nodes(self, nodes: list[RoadMapNode]) -> None:
        """
        Adds a list of nodes to the road map.
        :param nodes: given list of nodes to be added
        """
        for node in nodes:
            self.add_node(node=node)

    def remove_node(self, node_uid: str) -> None:
        """
        Removes a node from the road map
        :param node_uid: uid of the node to be removed
        """
        # first delete edges
        for to_uid in self.get()[node_uid].to_node_uid_set:
            # delete (node -> other)
            self.get()[to_uid].from_node_uid_set.remove(node_uid)
        for from_uid in self.get()[node_uid].from_node_uid_set:
            # delete (other -> node)
            self.get()[from_uid].to_node_uid_set.remove(node_uid)
        # then delete node
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

    def add_edges(self, edges: list[list[str]]) -> None:
        """
        Adds a list of edges.
        :param edges: a list of edges, with each edge represented as `[from_uid, to_uid]`
        """
        for (from_uid, to_uid) in edges:
            self.add_edge(from_uid=from_uid, to_uid=to_uid)

    def remove_edge(self, from_uid: str, to_uid: str) -> None:
        """
        Removes an edge from one node to the other.
        :param from_uid: uid of the source node
        :param to_uid: uid of the destination node
        """
        self.get()[from_uid].to_node_uid_set.remove(to_uid)
        self.get()[to_uid].from_node_uid_set.remove(from_uid)
        self._modified = True

    def _update_dependent_vars(self) -> None:
        """
        Updates the variables dependent on the road map.
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
            self._update_dependent_vars()
        return self._sample_x

    def sample_y(self) -> list[float]:
        """
        Gets y coordinate ordered list of the sample points.
        :return: y coordinate ordered list of the sample points
        """
        if self._modified:
            self._update_dependent_vars()
        return self._sample_y

    def sample_uid(self) -> list[str]:
        """
        Gets node uid ordered list of the sample points.
        :return: node uid ordered list of the sample points
        """
        if self._modified:
            self._update_dependent_vars()
        return self._sample_uid

    def kd_tree(self) -> KDTree:
        """
        Gets KD Tree for the sample points.
        :return: sample KD Tree
        """
        if self._modified:
            self._update_dependent_vars()
        return self._kd_tree

    def draw_road_map(self, pause: bool = True) -> None:
        """
        Draws the nodes and edges of the road map.
        :param pause: whether to pause `plt` a bit for rendering
        """
        # edges
        for (ix, iy, i_uid) in zip(self.sample_x(), self.sample_y(), self.sample_uid()):
            for j_uid in self.get()[i_uid].to_node_uid_set:
                j_node = self.get()[j_uid]
                plt.plot([ix, j_node.x],
                         [iy, j_node.y], '-y', alpha=0.2)  # -k = yellow solid line
        # nodes
        plt.plot(self.sample_x(), self.sample_y(), '.c')  # .c = cyan points

        if pause:
            plt.pause(0.001)

    def __len__(self):
        return len(self._road_map)

    def __str__(self):
        return self._road_map.__str__()
