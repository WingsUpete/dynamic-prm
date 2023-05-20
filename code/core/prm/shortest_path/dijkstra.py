from typing import Optional

import matplotlib.pyplot as plt

from core.util import Node2D

__all__ = ['dijkstra']


class DijkstraNode(Node2D):
    """ Node class for Dijkstra """
    def __init__(self, x: float, y: float, cost: float = 0, parent_id: int = -1):
        """
        Node class for Dijkstra.
        :param x: x coordinate
        :param y: y coordinate
        :param cost: cost from start to here
        :param parent_id: ID of parent node
        """
        # 2D Coordinates (float) & My Node ID (str)
        super().__init__(x=x, y=y)
        self.cost = cost
        self.parent_id = parent_id

    def __str__(self):
        return f'({self.x}, {self.y}, p={self.parent_id})'

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.node_id}: ({self.x}, {self.y}, p={self.parent_id})>'


def dijkstra(sample_x: list[float], sample_y: list[float], road_map: list[list[int]],
             start_id: int, end_id: int,
             animation: bool = True, animate_interval: int = 2) -> (Optional[list[list[float]]], float):
    """
    Runs Dijkstra algorithm to find the shortest path from starting point to end point, given the sample points + road
    map from PRM solver. Note that both starting point and end point are sample points.

    Reference: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/ProbabilisticRoadMap/probabilistic_road_map.py#L136
    :param sample_x: x coordinate list of the sample points
    :param sample_y: y coordinate list of the sample points
    :param road_map: road map specifying the edges formed among the sample points
    :param start_id: id of the starting sample point
    :param end_id: id of the end sample point
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

    start_node = DijkstraNode(x=sample_x[start_id], y=sample_y[start_id], cost=0)

    open_set, closed_set = {}, {}
    open_set[start_id] = start_node     # id -> node

    while True:
        if not open_set:
            return None, -1

        # pick the node from the open set with the smallest cost
        cur_node_id: int = min(open_set, key=lambda nid: open_set[nid].cost)
        current_node = open_set[cur_node_id]

        # animate searched points
        if animation and len(closed_set) % animate_interval == 0:
            plt.plot(current_node.x, current_node.y, 'xg')  # xg = green x marker
            plt.pause(0.001)

        # goal check
        if cur_node_id == end_id:
            # Get final path and calculate cost
            path = []
            cost = 0.0
            cur_node: DijkstraNode = current_node
            while True:
                path.append([cur_node.x, cur_node.y])
                if cur_node.parent_id == -1:
                    break

                parent_node: DijkstraNode = closed_set[cur_node.parent_id]
                cost += parent_node.euclidean_distance(other=cur_node)
                cur_node = parent_node

            path.reverse()
            return path, cost

        # Move current node from open set to closed set
        del open_set[cur_node_id]
        closed_set[cur_node_id] = current_node

        # Search and update neighbors
        for neighbor_node_id in road_map[cur_node_id]:
            if neighbor_node_id in closed_set:
                # already examined, skip
                continue

            neighbor_node = DijkstraNode(x=sample_x[neighbor_node_id], y=sample_y[neighbor_node_id])
            d = current_node.euclidean_distance(other=neighbor_node)
            neighbor_node.cost = current_node.cost + d
            neighbor_node.parent_id = cur_node_id

            if neighbor_node_id in open_set:
                # already visited but not yet examined
                if neighbor_node.cost < open_set[neighbor_node_id].cost:
                    # smaller cost, replace the cost of that node in the open set
                    open_set[neighbor_node_id].cost = neighbor_node.cost
                    open_set[neighbor_node_id].parent_id = cur_node_id
            else:
                # not yet visited, add it to the open set
                open_set[neighbor_node_id] = neighbor_node
