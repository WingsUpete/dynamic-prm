from datetime import datetime
from hashlib import sha256

__all__ = ['Node2D']


class Node2D:
    def __init__(self, x: float, y: float, node_id: str = None):
        # 2D Coordinates (float)
        self.x = x
        self.y = y

        # My Node ID (str)
        self.node_id = self._generate_id() if node_id is None else node_id

    def _generate_id(self) -> str:
        """
        Generates an identification string using SHA256. The semantic string for hashing includes the coordinates,
        as well as the current timestamp.
        :return: sha256 hash string as id
        """
        semantic_str = f'{self.x}{self.y}' + datetime.now().strftime('%Y%m%d %H:%M:%S')
        hash_str = sha256(semantic_str.encode()).hexdigest()
        return hash_str

    def __eq__(self, other):
        if (self.x == other.x) and (self.y == other.y):
            return True
        else:
            return False

    def __str__(self):
        return f'({self.x}, {self.y})'

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.node_id}: ({self.x}, {self.y})>'
