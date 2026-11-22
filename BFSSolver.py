from Node import Node
from Solver import Solver
from typing import Literal, cast
import numpy as np
from collections import deque

DEBUG = False

GOAL = b'G'
HOLE = b'H'

class BFSSolver(Solver):
    
    map: np.ndarray
    map_height: int
    map_width: int
    actions: list[Literal[0, 1, 2, 3]]

    # convert position to x and y coordinates
    def get_position(self):
        position = self.gym_env.unwrapped.s
        x = int(position % self.map_width)
        y = int(position / self.map_width)
        return x, y

    def __init__(self, gym_env):
        super().__init__(gym_env)
        self.nodes = {}
        self.nodes_expanded = 0
        self.map = self.gym_env.unwrapped.desc
        self.map_height, self.map_width = self.map.shape
        if DEBUG: print("BFSSolver initialized with map:")
        if DEBUG: print("map height:", self.map_height)
        if DEBUG: print("map width:", self.map_width)
        if DEBUG: print("initial position x and y:", self.get_position())
        if DEBUG: print(self.map)
        if DEBUG: print("solving...")
        solution = self.BFS()
        if solution is None:
            raise ValueError("No path to goal found!")
        if DEBUG: print("path:")
        if DEBUG: print(self.get_path_to_node(solution))
        if DEBUG: print("actions:")
        self.actions = self.path_to_actions(self.get_path_to_node(solution))
        if DEBUG: print(self.actions)
        if DEBUG: print("BFSSolver: nodes expanded:", self.nodes_expanded)

    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        if len(self.actions) > 0:
            action = self.actions.pop(0)
            return cast(Literal[0, 1, 2, 3], action)
        raise EOFError("No actions!")

    def path_to_actions(self, path):
        actions = []
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            if next_node.x == current_node.x + 1:
                actions.append(2)  # right
            elif next_node.x == current_node.x - 1:
                actions.append(0)  # left
            elif next_node.y == current_node.y + 1:
                actions.append(1)  # down
            elif next_node.y == current_node.y - 1:
                actions.append(3)  # up
        return actions

    def get_or_create_node(self, x, y, parent=None):
        if (x, y) not in self.nodes:
            self.nodes[(x, y)] = Node(x, y, parent=parent)
        return self.nodes[(x, y)]

    def get_path_to_node(self, node):
        path = []
        parent = node
        while parent is not None:
            path.append(parent)
            parent = parent.parent
        path.reverse()
        return path

    def is_goal(self, node):
        return self.map[node.y][node.x] == GOAL

    def is_hole(self, node):
        return self.map[node.y][node.x] == HOLE

    def get_neighbors(self, node):
        neighbors = []
        if node.x + 1 < self.map_width:
            neighbors.append(self.get_or_create_node(node.x+1, node.y, parent=node))
        if node.x - 1 >= 0:
            neighbors.append(self.get_or_create_node(node.x-1, node.y, parent=node))
        if node.y + 1 < self.map_height:
            neighbors.append(self.get_or_create_node(node.x, node.y+1, parent=node))
        if node.y - 1 >= 0:
            neighbors.append(self.get_or_create_node(node.x, node.y-1, parent=node))
        return neighbors

    def expand(self, node):
        self.nodes_expanded += 1
        children = []
        for n in self.get_neighbors(node):
            if self.is_hole(n) == False:
                children.append(n)
        return children
    
    def BFS(self):
        self.nodes.clear()
        frontier = deque()
        reached = set()
        node = self.get_or_create_node(*self.get_position())
        if self.is_goal(node):
            return node
        frontier.append(node)
        reached.add(node)
        while len(frontier) > 0:
            node = frontier.popleft()
            for child in self.expand(node):
                if self.is_goal(child):
                    return child
                if child not in reached:
                    reached.add(child)
                    frontier.append(child)
        return None


