from typing import List, Tuple

from tools.Point import Point


def flip_node_coordinates(list_of_nodes_yx):
    return [[yx[1], yx[0]] for yx in list_of_nodes_yx]


def flip_edge_coordinates(list_of_edges):
    return [flip_node_coordinates(edge_yx) for edge_yx in list_of_edges]


class Edge:
    def __init__(self, current_point: Point = None):
        self._points = []

        self._current_point = current_point

        self._reached: bool = False
        self.neighbour_is_node: bool = False

    def __repr__(self):
        if len(self) == 0:
            arr_str = ""
        elif len(self) == 1:
            arr_str = f"{self._points[0]}"
        elif len(self) <= 3:
            elems = [f"{p}" for p in self._points]
            arr_str = ", ".join(elems)
        else:
            elems = [f"{p}" for p in self._points[:3]] + [
                "... ",
                repr(self._points[-1]),
            ]
            arr_str = ", ".join(elems)

        return f"[{arr_str}]"

    def __len__(self):
        return len(self._points)

    def add(self, point: Point):
        self._points.append(point)

    def sort(self):
        edge_start, edge_end = self._points[0], self._points[-1]

        # is_circular = edge_start == edge_end

        start_col_greater = edge_start.col > edge_end.col
        start_col_same = edge_start.col == edge_end.col
        start_row_smaller = edge_start.row < edge_end.row

        if start_col_greater or (start_col_same and start_row_smaller):
            self._points.reverse()

        # if is_circular:
        #     start_2, end_2 = self._points[1], self._points[-2]
        #
        #     start_row_greater = start_2.row > end_2.row
        #     start_row_same = start_2.row == end_2.row
        #     start_col_smaller = start_2.col < end_2.col
        #
        #     if start_row_greater or (start_row_same and start_col_smaller):
        #         self._points.reverse()

    @property
    def current_point(self):
        return self._current_point

    @current_point.setter
    def current_point(self, value):
        self._current_point = value

    @property
    def reached(self) -> bool:
        return self._reached

    @reached.setter
    def reached(self, value: bool) -> None:
        self._reached = value
        self.sort()

    @property
    def points(self) -> List[Point]:
        return self._points.copy()

    @property
    def points_yx(self) -> List[List[int]]:
        return [[p.row, p.col] for p in self._points]

    @property
    def points_xy(self) -> List[List[int]]:
        return [[p.col, p.row] for p in self._points]

    @property
    def start_end(self) -> Tuple[Point, Point]:
        return self.points[0], self.points[-1]

    @property
    def start_end_yx(self) -> List[List[int]]:
        s, e = self.start_end
        return [[s.row, s.col], [e.row, e.col]]

    @property
    def start_end_xy(self) -> List[List[int]]:
        s, e = self.start_end
        return [[s.col, s.row], [e.col, e.row]]
