import os
import sys
sys.path.append(os.path.join(os.getcwd(), '../../'))  # load core
from random import random

import matplotlib.pyplot as plt

from core.prm import *


def print_roadmap_with_from_edges(rmp: RoadMap) -> None:
    print(rmp.get().keys())
    for node in rmp.get().values():
        for from_uid in node.from_node_uid_dict.keys():
            print(f'{from_uid} --> {node.node_uid}')


def print_roadmap_with_to_edges(rmp: RoadMap) -> None:
    print(rmp.get().keys())
    for node in rmp.get().values():
        for to_uid in node.to_node_uid_dict.keys():
            print(f'{node.node_uid} --> {to_uid}')


def construct_roadmap() -> RoadMap:
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
    return rmp


def test_roadmap():
    rmp = construct_roadmap()
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


def test_blocking():
    rmp = construct_roadmap()
    print_roadmap_with_to_edges(rmp)
    rmp.draw_road_map()
    plt.pause(0.001)
    plt.waitforbuttonpress()
    print('---------------------------')
    print('Block "2 --> 4"')
    rmp.block_edge(from_uid='2', to_uid='4')
    print_roadmap_with_to_edges(rmp.get_clear_roadmap())
    plt.clf()
    rmp.draw_road_map()
    plt.pause(0.001)
    plt.waitforbuttonpress()
    print('Recover')
    rmp.unblock_edge(from_uid='2', to_uid='4')
    print_roadmap_with_to_edges(rmp.get_clear_roadmap())
    plt.clf()
    rmp.draw_road_map()
    plt.pause(0.001)
    plt.waitforbuttonpress()
    print('---------------------------')
    print('Block node 5')
    rmp.block_node(node_uid='5')
    print_roadmap_with_to_edges(rmp.get_clear_roadmap())
    plt.clf()
    rmp.draw_road_map()
    plt.pause(0.001)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    # test_roadmap()
    test_blocking()
