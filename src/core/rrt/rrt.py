import math
import random
import time
from typing import Optional

import matplotlib.pyplot as plt

from core.util import Node2D, plot_circle

__all__ = ['Rrt', 'RrtNode']


class RrtNode(Node2D):
    def __init__(self, x: float, y: float):
        super().__init__(x=x, y=y)
        self.path_x: list[float] = []
        self.path_y: list[float] = []
        self.parent: Optional[RrtNode] = None

    def get_path_with_cost(self) -> (list[list[float]], float):
        """
        Gets the path to current node
        :return: path
        """
        path = []
        cost = 0.0
        cur_node = self
        while True:
            path.append([cur_node.x, cur_node.y])
            if cur_node.parent is None:
                break

            cost += cur_node.parent.euclidean_distance(other=cur_node)
            cur_node = cur_node.parent

        path.reverse()
        return path, cost


class Rrt:
    """
    RRT: Rapidly-exploring Random Tree

    Reference: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
    """
    def __init__(self,
                 map_range: list[float], obstacles: list[list[float]],
                 robot_radius: float, start: list[float], goal: list[float],
                 goal_sample_rate: float = 0.05, expand_dis: float = 3.0, path_resolution: float = 0.5,
                 max_iter: int = 500, rnd_seed: int = None,
                 animate_interval: int = 5, pause_interval: Optional[float] = None):
        """
        Creates an RRT Solver to solve the given problem.
        :param map_range: the range of the map, as `[min, max]` for both `x` and `y`
        :param obstacles: list of circle obstacles, each represented as `[x, y, r]`
        :param robot_radius: radius of the circle robot
        :param start: starting position for the problem
        :param goal: goal position for the problem
        :param goal_sample_rate: specifies how frequent should the algorithm sample the goal point for analysis
        :param expand_dis: expand at most x from current node (for steering function)
        :param path_resolution: specifies how long a maximum straight-line unit-path is (can go `x` at most at a time)
        :param max_iter: maximum number of iterations for the algorithm
        :param rnd_seed: random seed for sampler
        :param animate_interval: specifies how frequent (every x steps) should the graph be rendered
        :param pause_interval: specifies how long (sec) should the program wait til the next step starts
        """
        self.map_min, self.map_max = map_range
        self.obstacles = obstacles

        self.robot_radius = robot_radius
        self.start = RrtNode(x=start[0], y=start[1])
        self.goal = RrtNode(x=goal[0], y=goal[1])

        self.goal_sample_rate = goal_sample_rate
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution

        self.max_iter = max_iter
        self.rnd_seed = rnd_seed
        random.seed(self.rnd_seed)

        self.animate_interval = animate_interval
        self.pause_interval = pause_interval

        self.node_list: list[RrtNode] = []

    def plan(self, animation=True) -> (Optional[list[list[float]]], float, int):
        """
        Plans the route using RRT.
        :param animation: specifies whether to enable real-time animation
        :return: found feasible path as an ordered list of 2D points, or None if not found + path cost
        + how many steps are iterated
        """
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # Sample one node on the map
            rnd_node = self.get_random_node()
            # Get nearest neighbor of the sampled node
            nearest_node = self.get_nearest_node(rnd_node)

            # Get the expected new node using steering function (how far `nearest_node` can reach along the
            # direction to the `rnd_node`)
            new_node = self.steer(from_node=nearest_node, to_node=rnd_node)

            # check collision for `new_node`
            if self.pass_collision_check(new_node):
                self.node_list.append(new_node)
                # goal check
                if new_node.euclidean_distance(self.goal) <= self.expand_dis:
                    final_node = self.steer(from_node=new_node, to_node=self.goal)
                    if self.pass_collision_check(final_node):
                        self.node_list.append(final_node)
                        final_path, final_path_cost = final_node.get_path_with_cost()
                        return final_path, final_path_cost, i + 1

            # render graph
            if animation and i % self.animate_interval == 0:
                self.draw_graph(rnd_node=rnd_node, new_node=new_node)
                if self.pause_interval:
                    time.sleep(self.pause_interval)

        # path not found within given steps
        return None, -1, self.max_iter

    def get_random_node(self) -> RrtNode:
        """
        Randomly samples a node on the map, with a probability of sampling the goal point directly
        :return: the sampled node.
        """
        if random.uniform(0, 1) <= self.goal_sample_rate:
            # sample goal point
            rnd = RrtNode(x=self.goal.x, y=self.goal.y)
        else:
            # sample random point on map
            rnd = RrtNode(
                x=random.uniform(self.map_min, self.map_max),
                y=random.uniform(self.map_min, self.map_max),
            )
        return rnd

    def get_nearest_node(self, node: RrtNode) -> RrtNode:
        """
        Retrieves the nearest node in the node list to the provided node.

        # TODO: too time consuming?
        :return: the nearest node
        """
        dist_list = [node.euclidean_squared_distance(cur_node) for cur_node in self.node_list]
        nearest_ind = dist_list.index(min(dist_list))
        nearest_node = self.node_list[nearest_ind]
        return nearest_node

    def steer(self, from_node: RrtNode, to_node: RrtNode) -> RrtNode:
        """
        Steers the path from one node to the other, and retrieve a new node to replace the `to_node`. Intuitively, it
        tries to start from `from_node` and see how far it can reach along the direction to the `to_node`, given a
        maximum `expand_dis` (expand distance) and the path resolution (how long a maximum unit-path is).
        :param from_node: from this node
        :param to_node: to this node
        :return: a new node provided by the steering function to replace the `to_node`
        """
        new_node = RrtNode(x=from_node.x, y=from_node.y)
        d, theta = self.cal_dist_n_angle(from_node=new_node, to_node=to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        # go for `extend_length` at most (if `d` is shorter, then just go for `d`)
        extend_length = self.expand_dis if self.expand_dis <= d else d
        # need to go `n_expand` times before achieving the `extend_length`, but cannot exceed `extend_length`
        n_expand = math.floor(extend_length / self.path_resolution)
        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        # expected destination
        expected_to_node = RrtNode(x=(from_node.x + extend_length * math.cos(theta)),
                                   y=(from_node.y + extend_length * math.sin(theta)))
        d, _ = self.cal_dist_n_angle(from_node=new_node, to_node=expected_to_node)
        if 0 < d <= self.path_resolution:
            # d > 0: still some distance
            # d <= path_resolution: can reach `expected_to_node` with one more shorter sub-path
            new_node.path_x.append(expected_to_node.x)
            new_node.path_y.append(expected_to_node.y)
            new_node.x = expected_to_node.x
            new_node.y = expected_to_node.y

        new_node.parent = from_node

        return new_node

    @staticmethod
    def cal_dist_n_angle(from_node: RrtNode, to_node: RrtNode) -> (float, float):
        """
        Calculates the distance and angle from one node to the other.
        :param from_node: from this node
        :param to_node: to this node
        :return: `d` as distance, `theta` as angle
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def pass_collision_check(self, node: Optional[RrtNode]) -> bool:
        """
        Checks whether the robot at the given node collides with any obstacle. If collision check passes, return True!
        :param node: given node specifying where the robot is standing
        :return: collision check passes or not (if node is null, return False)
        """
        for (ox, oy, o_radius) in self.obstacles:
            for path_i in range(len(node.path_x)):
                squared_dist = RrtNode(x=node.path_x[path_i],
                                       y=node.path_y[path_i]).euclidean_squared_distance(other=RrtNode(x=ox,
                                                                                                       y=oy))
                if squared_dist <= (self.robot_radius + o_radius)**2:
                    return False

        return True

    def draw_graph(self,
                   rnd_node: Optional[RrtNode] = None,
                   new_node: Optional[RrtNode] = None,
                   path: list[list[float]] = None) -> None:
        """
        Re-renders the whole graph. More details:

        > 1. Obstacles: black circles

        > 2. Starting point: red x marker

        > 3. Goal point: red thin_diamond marker

        > 4. Examined paths: green solid lines.

        > 5. Sampled `rnd_node`: cyan triangle

        > 6. Steered `new_node`: magenta circle

        > 7. Specified path: blue solid line

        :param rnd_node: sampled `rnd_node`
        :param new_node: new node steered from `rnd_node`
        :param path: a specific path to render
        """
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None]
        )

        # set global map info
        plt.axis('equal')
        plt.xlim([self.map_min, self.map_max])
        plt.ylim([self.map_min, self.map_max])
        plt.grid(False)

        # draw all obstacles
        for (ox, oy, o_radius) in self.obstacles:
            plot_circle(ox, oy, o_radius, 'k', fill=True, pause=False)    # -k = black solid line ---> a big black circle

        # draw all paths for nodes in the node list
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, '-g')    # -g = green solid line

        # draw `rnd_node` and `new_node`
        if rnd_node is not None and new_node is not None:
            plt.plot(rnd_node.x, rnd_node.y, '^c')  # ^c = cyan triangle
            if self.robot_radius > 0.0:
                plot_circle(new_node.x, new_node.y, self.robot_radius, 'm', fill=False, pause=False)   # m = magenta

        # draw the specified path
        if path:
            plt.plot([x for (x, _) in path], [y for (_, y) in path], '-b')  # -b = blue solid line

        # draw start & goal
        plt.plot(self.start.x, self.start.y, 'xr')  # xr = red x marker
        plt.plot(self.goal.x, self.goal.y, 'dr')  # dr = red thin_diamond marker

        plt.pause(0.01)
