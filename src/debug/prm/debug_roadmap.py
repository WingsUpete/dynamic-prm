import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../'))  # load core
from random import random

from core.prm import *


def print_roadmap_with_from_edges(rmp: RoadMap) -> None:
    print(rmp.get().keys())
    for node in rmp.get().values():
        for from_uid in node.from_node_uid_set:
            print(f'{from_uid} --> {node.node_uid}')


def print_roadmap_with_to_edges(rmp: RoadMap) -> None:
    print(rmp.get().keys())
    for node in rmp.get().values():
        for to_uid in node.to_node_uid_set:
            print(f'{node.node_uid} --> {to_uid}')


def test_roadmap():
    rmp = RoadMap()
    for i in range(7):
        rmp.add_node(node=RoadMapNode(x=random(), y=random(), node_uid=f'{i+1}'))
    rmp.add_edges(edges=[
        ['1', '2'],
        ['1', '3'],
        ['2', '4'],
        ['4', '5'],
        ['5', '6'],
        ['5', '7'],
        ['7', '6'],
    ])
    print('from_edges:')
    print_roadmap_with_from_edges(rmp)
    print('---------------------------')
    print('to_edges:')
    print_roadmap_with_to_edges(rmp)
    print('---------------------------')
    print('---------------------------')
    print('---------------------------')
    print('Remove "2 --> 4"')
    rmp.remove_edge(from_uid='2', to_uid='4')
    print_roadmap_with_to_edges(rmp)
    print('---------------------------')
    print('---------------------------')
    print('Remove node 5')
    rmp.remove_node(node_uid='5')
    print_roadmap_with_to_edges(rmp)


if __name__ == '__main__':
    test_roadmap()
