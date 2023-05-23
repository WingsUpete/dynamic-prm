from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

from core.util.common import *
from core.util.graph import Node2D

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

        # node is clear (not blocked)
        self.clear = True

        # storing neighbors: uid -> road is clear (not blocked) for this edge
        self.from_node_uid_dict: dict[str, bool] = {}
        self.to_node_uid_dict: dict[str, bool] = {}


class RoadMap:
    """ Road Map """
    def __init__(self, enable_kd_tree: bool = True):
        """
        Creates a Road Map that stores the sample point graph as well as a KD Tree for sample points.
        :param enable_kd_tree: specifies whether to enable KD Tree.
        """
        # node_uid -> node
        self._road_map: OrderedDict[str, RoadMapNode] = OrderedDict()

        # ordered list of separated sample coordinates transformed from the road map
        self._sample_x: list[float] = []
        self._sample_y: list[float] = []

        # ordered list of sample node uid transformed from the road map
        self._sample_uid: list[str] = []

        # kd tree for sample points
        self._enable_kd_tree = enable_kd_tree
        self._kd_tree: Optional[KDTree] = None

        # if road map modified, need to update other variables
        self._modified = True

    def get(self):
        """
        Gets the road map.
        :return: road map
        """
        return self._road_map

    def get_clear_roadmap(self) -> RoadMap:
        """
        Gets a new version of road map with all blocked nodes/edges deleted
        :return: a new road map
        """
        new_road_map = RoadMap(enable_kd_tree=self.kd_tree_enabled())

        # add clear nodes
        new_clear_nodes = [RoadMapNode(
            x=cur_node.x, y=cur_node.y, node_uid=cur_node.node_uid
        ) for cur_node in self.get().values() if cur_node.clear]
        new_road_map.add_nodes(nodes=new_clear_nodes)

        # add clear edges
        for cur_uid in new_road_map.get().keys():
            for (to_uid, to_able) in self.get()[cur_uid].to_node_uid_dict.items():
                if to_able:
                    new_road_map.add_edge(from_uid=cur_uid, to_uid=to_uid)

        return new_road_map

    def get_node_by_index(self, index: int) -> RoadMapNode:
        """
        Retrieves the node according to the node index on the ordered list transformed from the road map
        :param index: node index on the ordered list
        :return: the corresponding node
        """
        return self.get()[self.sample_uid()[index]]

    def get_knn(self, point: list[float], k: int) -> (list[float], list[int]):
        """
        Gets `k` nearest neighbors using KD Tree. Returns nothing if KD Tree is disabled.
        :param point: given point to be examined
        :param k: number of nearest neighbors to get
        :return: `k` nearest neighbors (distances, indices of neighbor nodes), or nothing if KD Tree is disabled
        """
        if not self._enable_kd_tree:
            return [], []
        dists, indices = self.kd_tree().query(point, k=k)
        dists = [dists] if type(dists) == float else dists
        indices = [indices] if type(indices) == int else indices
        return dists, indices

    def get_nearest_neighbor(self, point: list[float]) -> (float, int):
        """
        Gets the nearest neighbor (TO the point). Uses KD Tree by default, otherwise use brute force.
        :param point: give point to be examined
        :return: the nearest neighbor (distance to it, index of it in the road map as ordered list)
        """
        if self._enable_kd_tree:
            return self.kd_tree().query(point)
        else:
            dist_list = [cal_dist(from_x=cur_node.x, from_y=cur_node.y,
                                  to_x=point[0], to_y=point[1]) for cur_node in self.get().values()]
            nearest_ind = dist_list.index(min(dist_list))
            return nearest_ind

    def find_points_within_r(self, point: list[float], r: float) -> list[int]:
        """
        Finds all points within distance `r` of the given point.
        :param point: given point to be examined
        :param r: radius `r`
        :return: all satisfied points as a list of indices, or nothing if KD Tree is disabled
        """
        if not self._enable_kd_tree:
            return []
        indices = self.kd_tree().query_ball_point(point, r=r)
        indices = [indices] if type(indices) == int else indices
        return indices

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
        for to_uid in self.get()[node_uid].to_node_uid_dict.keys():
            # delete (node -> other)
            del self.get()[to_uid].from_node_uid_dict[node_uid]
        for from_uid in self.get()[node_uid].from_node_uid_dict.keys():
            # delete (other -> node)
            del self.get()[from_uid].to_node_uid_dict[node_uid]
        # then delete node
        del self.get()[node_uid]
        self._modified = True

    def block_node(self, node_uid: str) -> None:
        """
        Blocks a node due to added obstacles. Note that all related edges are also blocked.
        :param node_uid: uid of the node
        """
        # block edges
        for to_uid in self.get()[node_uid].to_node_uid_dict.keys():
            self.get()[node_uid].to_node_uid_dict[to_uid] = False
            self.get()[to_uid].from_node_uid_dict[node_uid] = False
        for from_uid in self.get()[node_uid].from_node_uid_dict.keys():
            self.get()[node_uid].from_node_uid_dict[from_uid] = False
            self.get()[from_uid].to_node_uid_dict[node_uid] = False

        # block node
        self.get()[node_uid].clear = False

    def unblock_node(self, node_uid: str) -> None:
        """
        Unblocks a node due to removed obstacles. Note that all related edges should be checked for connectivity.

        TODO: current implementation is not correct. Should move it to PRM later.
        :param node_uid: uid of the node
        """
        # unblock edges
        for to_uid in self.get()[node_uid].to_node_uid_dict.keys():
            self.get()[node_uid].to_node_uid_dict[to_uid] = True
            self.get()[to_uid].from_node_uid_dict[node_uid] = True
        for from_uid in self.get()[node_uid].from_node_uid_dict.keys():
            self.get()[node_uid].from_node_uid_dict[from_uid] = True
            self.get()[from_uid].to_node_uid_dict[node_uid] = True

        # unblock node
        self.get()[node_uid].clear = True

    def add_edge(self, from_uid: str, to_uid: str) -> None:
        """
        Adds an edge from one node to the other.
        :param from_uid: uid of the source node
        :param to_uid: uid of the destination node
        """
        self.get()[from_uid].to_node_uid_dict[to_uid] = True
        self.get()[to_uid].from_node_uid_dict[from_uid] = True
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
        del self.get()[from_uid].to_node_uid_dict[to_uid]
        del self.get()[to_uid].from_node_uid_dict[from_uid]
        self._modified = True

    def block_edge(self, from_uid: str, to_uid: str) -> None:
        """
        Blocks an edge due to added obstacles.
        :param from_uid: uid of the source node
        :param to_uid: uid of the destination node
        """
        self.get()[from_uid].to_node_uid_dict[to_uid] = False
        self.get()[to_uid].from_node_uid_dict[from_uid] = False

    def unblock_edge(self, from_uid: str, to_uid: str) -> None:
        """
        Unblocks an edge due to removed obstacles.
        :param from_uid: uid of the source node
        :param to_uid: uid of the destination node
        """
        self.get()[from_uid].to_node_uid_dict[to_uid] = True
        self.get()[to_uid].from_node_uid_dict[from_uid] = True

    def enable_kd_tree(self) -> None:
        """
        Enables the KD Tree.
        """
        self._enable_kd_tree = True
        self._modified = True

    def disable_kd_tree(self) -> None:
        """
        Disables the KD Tree.
        """
        self._enable_kd_tree = False

    def _update_dependent_vars(self) -> None:
        """
        Updates the variables dependent on the road map.
        """
        self._sample_x = [node.x for node in self.get().values()]
        self._sample_y = [node.y for node in self.get().values()]
        self._sample_uid = [node.node_uid for node in self.get().values()]
        if self._enable_kd_tree:
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

    def kd_tree_enabled(self) -> bool:
        """
        Checks whether KD Tree is enabled.
        :return: True if enabled
        """
        return self._enable_kd_tree

    def kd_tree(self) -> Optional[KDTree]:
        """
        Gets KD Tree for the sample points.
        :return: sample KD Tree
        """
        if not self._enable_kd_tree:
            return None
        if self._modified:
            self._update_dependent_vars()
        return self._kd_tree

    def draw_road_map(self) -> None:
        """
        Draws the nodes and edges of the road map.
        """
        # edges
        for (ix, iy, i_uid) in zip(self.sample_x(), self.sample_y(), self.sample_uid()):
            for (j_uid, ij_clear) in self.get()[i_uid].to_node_uid_dict.items():
                j_node = self.get()[j_uid]
                plt.plot([ix, j_node.x],
                         [iy, j_node.y],
                         '-y' if ij_clear else '-r',
                         alpha=0.2)  # -y = yellow solid line (-r = red solid line)
        # nodes
        clear_node_coords = [(cur_node.x, cur_node.y) for cur_node in self.get().values() if cur_node.clear]
        blocked_node_coords = [(cur_node.x, cur_node.y) for cur_node in self.get().values() if not cur_node.clear]
        plt.plot([x for (x, _) in clear_node_coords], [y for (_, y) in clear_node_coords], '.c')  # .c = cyan points
        plt.plot([x for (x, _) in blocked_node_coords], [y for (_, y) in blocked_node_coords], '.r')  # .r = red points

    def __len__(self):
        return len(self._road_map)

    def __str__(self):
        return self._road_map.__str__()
