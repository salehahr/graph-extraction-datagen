from __future__ import annotations

from typing import List, Union

import numpy as np

from tools.NodeType import NodeType


def all_neighbours(middlepoint: Union[List[int], np.ndarray]):
    # list of all pixels in neihgbourhood of [a,b]
    a, b = middlepoint[0], middlepoint[1]
    return [[xx, yy] for xx in range(a - 1, a + 2) for yy in range(b - 1, b + 2)]


def positive_neighbours(a: int, b: int, image: np.ndarray) -> List[List[int]]:
    # list of pixels with value 1 in neighbourhood of [a,b]
    nb = [
        [xx, yy]
        for xx in range(a - 1, a + 2)
        for yy in range(b - 1, b + 2)
        if image[xx, yy] == 1
    ]
    if [a, b] in nb:
        nb.remove([a, b])

    return nb


def get_sorted_neighbours(
    point: Union[np.ndarray, List[int]], img: np.ndarray
) -> List[List[int]]:
    """Nachbarn abhängig von Distanz sortieren."""

    n = positive_neighbours(point[0], point[1], img)

    if len(n) > 1:
        dist = [[distance(point, neighb), neighb] for neighb in n]
        dist_sorted = sorted(dist, key=lambda x: x[0])
        n = [neighb for _, neighb in dist_sorted]

    return n


def four_connectivity(*args) -> List[List[int]]:
    """Returns list of pixels in 4-connectivity of [a, b]."""

    if len(args) == 1:
        a, b = args[0]
    elif len(args) == 2:
        a, b = args[0], args[1]
    else:
        raise Exception

    return [[a + 1, b], [a - 1, b], [a, b + 1], [a, b - 1]]


def num_in_4connectivity(a: int, b: int, image: np.ndarray):
    """How many pixel with value 255 are in 4-connectivity of [a, b]."""
    neighbours = four_connectivity(a, b)
    max_val = 1 if np.max(image) == 1 else 255
    pos_vals = [1 for (nr, nc) in neighbours if image[nr, nc] == max_val]
    return len(pos_vals)


def distance(a: Union[List[int], np.ndarray], b: Union[List[int], np.ndarray]) -> float:
    if isinstance(a, list) or isinstance(b, list):
        a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)


def remove_cross_nodes(points: List[Point]) -> List[Point]:
    """Removes nodes from list if they are crossing nodes."""
    points = points.copy()
    nodes_to_delete = [p for p in points if p.type == NodeType.CROSSING]
    for n in nodes_to_delete:
        points.remove(n)
    return points


class Point:
    def __init__(self, *args, node_type: NodeType = None):
        if isinstance(args, np.ndarray) or len(args) == 2:
            row, col = args[0], args[1]
        elif len(args) == 1:
            row, col = args[0]
        else:
            raise Exception

        self.row = row
        self.col = col
        self.type = node_type

        self._all_neighbours = []
        self._neighbours = []

    def __repr__(self):
        type_str = self.type.name if self.type is not None else "?"
        return f"P({self.row}, {self.col}) : {type_str}"

    def __eq__(self, other) -> bool:
        same_row = self.row == other.row
        same_col = self.col == other.col
        return same_row and same_col

    def value_in(self, img: np.ndarray) -> int:
        return img[self.row, self.col]

    def is_cross_node(self, img: np.ndarray) -> bool:
        cross = []
        row, col = self.row, self.col

        if img[row - 1, col - 1] == 1:
            cross.append(True)
        if img[row + 1, col - 2] == 1:
            cross.append(True)
        if img[row, col - 1] == 1:
            cross.append(True)
        if img[row - 1, col] == 1:
            cross.append(True)
        if img[row - 2, col - 2] == 1:
            cross.append(True)
        if img[row - 2, col + 1] == 1:
            cross.append(True)
        if img[row + 1, col + 1] == 1:
            cross.append(True)

        return len(cross) == 7

    def find_neighbours(self, img: np.ndarray, node_list: list = None) -> None:
        self._all_neighbours = [
            Point(row, col) for (row, col) in all_neighbours(self.arr)
        ]
        self._neighbours = [
            Point(row, col)
            for (row, col) in positive_neighbours(self.row, self.col, img)
        ]

        if node_list is not None:
            self._all_neighbours = [
                node_list[node_list.index(n)] if n in node_list else n
                for n in self._all_neighbours
            ]
            self._neighbours = [
                node_list[node_list.index(n)] if n in node_list else n
                for n in self._neighbours
            ]

    def find_dist_one_points(self, others: List[Point]):
        return [n for n in others if self.distance_to(n) == 1]

    def distance_to(self, other: Point) -> float:
        return distance(self.arr, other.arr)

    @property
    def arr(self) -> np.ndarray:
        return np.array([self.row, self.col])

    @property
    def list_rc(self) -> List[int]:
        return [self.row, self.col]

    @property
    def all_neighbours(self):
        return self._all_neighbours

    @property
    def neighbours(self):
        return self._neighbours

    @neighbours.setter
    def neighbours(self, value):
        raise Exception

    @property
    def sorted_neighbours(self) -> List[Point]:
        if self.num_neighbours > 1:
            dist = [[self.distance_to(n), n] for n in self._neighbours]
            return [n for _, n in sorted(dist, key=lambda x: x[0])]
        else:
            return self._neighbours

    @property
    def num_neighbours(self) -> int:
        return len(self._neighbours)

    @property
    def four_connectivity(self) -> List[Point]:
        return [Point(row, col) for (row, col) in four_connectivity(self.row, self.col)]
