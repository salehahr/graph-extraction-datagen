from __future__ import annotations

from typing import List, Union

import numpy as np


def all_neighbours(middlepoint: list):
    # list of all pixels in neihgbourhood of [a,b]
    a, b = middlepoint
    return [[xx, yy] for xx in range(a - 1, a + 2) for yy in range(b - 1, b + 2)]


def positive_neighbours(a: int, b: int, image: np.ndarray) -> List[List[int]]:
    # list of pixels with value 1 in in neighbourhood of [a,b]
    nb = [
        [xx, yy]
        for xx in range(a - 1, a + 2)
        for yy in range(b - 1, b + 2)
        if image[xx, yy] == 1
    ]
    if [a, b] in nb:
        nb.remove([a, b])

    return nb


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


def distance(
    a: Union[List[List[int]], np.ndarray], b: Union[List[List[int]], np.ndarray]
):
    if isinstance(a, list) or isinstance(b, list):
        a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)


class Point:
    def __init__(self, *args):
        if isinstance(args, np.ndarray) or len(args) == 2:
            row, col = args[0], args[1]
        elif len(args) == 1:
            row, col = args[0]
        else:
            raise Exception

        self.row = row
        self.col = col

        self._neighbours = []

    def __eq__(self, other) -> bool:
        same_row = self.row == other.row
        same_col = self.col == other.col
        return same_row and same_col

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

    def find_positive_neighbours(self, img: np.ndarray) -> None:
        self._neighbours = [
            Point(row, col)
            for (row, col) in positive_neighbours(self.row, self.col, img)
        ]

    def find_dist_one_points(self, others: List[Point]):
        return [n for n in others if self.distance_to(n) == 1]

    def distance_to(self, other: Point):
        return distance(self.arr, other.arr)

    @property
    def arr(self):
        return np.array([self.row, self.col])

    @property
    def neighbours(self):
        return self._neighbours

    @neighbours.setter
    def neighbours(self, value):
        raise Exception

    @property
    def num_neighbours(self) -> int:
        return len(self._neighbours)

    @property
    def four_connectivity(self) -> List[Point]:
        return [Point(row, col) for (row, col) in four_connectivity(self.row, self.col)]
