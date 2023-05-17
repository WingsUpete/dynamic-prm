import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../'))   # load core
from core.shortest_path import DijkstraNode


def test_node():
    node0 = DijkstraNode(x=10, y=20)
    print(node0)


if __name__ == '__main__':
    test_node()
