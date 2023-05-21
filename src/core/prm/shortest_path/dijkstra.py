from typing import Optional

import matplotlib.pyplot as plt

from core.prm import RoadMap
from core.util import Node2D

__all__ = ['dijkstra']


class DijkstraNode(Node2D):
    """ Node class for Dijkstra """
    def __init__(self, x: float, y: float, cost: float = 0, parent_uid: str = None):
        """
        Node class for Dijkstra.
        :param x: x coordinate
        :param y: y coordinate
        :param cost: cost from start to here
        :param parent_uid: uid of parent node
        """
        super().__init__(x=x, y=y)
        self.cost = cost
        self.parent_uid = parent_uid


def dijkstra(road_map: RoadMap, start_uid: str, end_uid: str,
             animation: bool = True, animate_interval: int = 2) -> (Optional[list[list[float]]], float):
    """
    Runs Dijkstra algorithm to find the shortest path from starting point to end point, given the sample points + road
    map from PRM solver. Note that both starting point and end point are sample points.

    Reference: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py#L136
    :param road_map: road map specifying the edges formed among the sample points
    :param start_uid: uid of the starting sample point
    :param end_uid: uid of the end sample point
    :param animation: enables animation or not
    :param animate_interval: specifies how frequent (every x new points added to closed set) should the searched
    points be rendered
    :return: found feasible path as an ordered list of 2D points, or None if not found + path cost
    """
    if animation:
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None]
        )

    start_road_map_node = road_map.get()[start_uid]
    start_node = DijkstraNode(x=start_road_map_node.x, y=start_road_map_node.y, cost=0)

    open_set, closed_set = {}, {}
    open_set[start_uid] = start_node     # uid -> node

    while True:
        if not open_set:
            return None, -1

        # pick the node from the open set with the smallest cost
        cur_node_uid: str = min(open_set, key=lambda n_uid: open_set[n_uid].cost)
        current_node = open_set[cur_node_uid]

        # animate searched points
        if animation and len(closed_set) % animate_interval == 0:
            plt.plot(current_node.x, current_node.y, 'xg')  # xg = green x marker
            plt.pause(0.001)

        # goal check
        if cur_node_uid == end_uid:
            # Get final path and calculate cost
            path = []
            cost = 0.0
            cur_node: DijkstraNode = current_node
            while True:
                path.append([cur_node.x, cur_node.y])
                if cur_node.parent_uid is None:
                    break

                parent_node: DijkstraNode = closed_set[cur_node.parent_uid]
                cost += parent_node.euclidean_distance(other=cur_node)
                cur_node = parent_node

            path.reverse()
            return path, cost

        # Move current node from open set to closed set
        del open_set[cur_node_uid]
        closed_set[cur_node_uid] = current_node

        # Search and update neighbors=
        for neighbor_node_uid in road_map.get()[cur_node_uid].to_node_uid_set:
            if neighbor_node_uid in closed_set:
                # already examined, skip
                continue

            neighbor_road_map_node = road_map.get()[neighbor_node_uid]
            neighbor_node = DijkstraNode(x=neighbor_road_map_node.x, y=neighbor_road_map_node.y)
            d = current_node.euclidean_distance(other=neighbor_node)
            neighbor_node.cost = current_node.cost + d
            neighbor_node.parent_uid = cur_node_uid

            if neighbor_node_uid in open_set:
                # already visited but not yet examined
                if neighbor_node.cost < open_set[neighbor_node_uid].cost:
                    # smaller cost, replace the cost of that node in the open set
                    open_set[neighbor_node_uid].cost = neighbor_node.cost
                    open_set[neighbor_node_uid].parent_uid = neighbor_node.parent_uid
            else:
                # not yet visited, add it to the open set
                open_set[neighbor_node_uid] = neighbor_node
