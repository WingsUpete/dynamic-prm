from core.util import Node2D

__all__ = ['DijkstraNode']


class DijkstraNode(Node2D):
    """ Node class for Dijkstra """
    def __init__(self, x: float, y: float, parent_id: str = None, node_id: str = None):
        """
        Node class for Dijkstra.
        :param x: x coordinate
        :param y: y coordinate
        :param parent_id: ID of parent node
        :param node_id: ID of current node, auto-generated if not provided
        """
        # 2D Coordinates (float) & My Node ID (str)
        super().__init__(x=x, y=y, node_id=node_id)

        # Parent Node ID (str)
        self.parent_id = parent_id

    def __str__(self):
        return f'({self.x}, {self.y}, p={self.parent_id})'

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.node_id}: ({self.x}, {self.y}, p={self.parent_id})>'
