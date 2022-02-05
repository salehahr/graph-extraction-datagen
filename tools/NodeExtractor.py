import numpy as np

from tools.images import normalise
from tools.NodeContainer import NodeContainer
from tools.Point import Point


class NodeExtractor:
    def __init__(self, img: np.ndarray):
        self.img = normalise(img)  # volatile
        self.orig_img = normalise(img)  # unchanging

        kernel = 3
        self.kn = int(np.floor(kernel / 2))
        height, width = img.shape

        self.row_indices = range(self.kn, height - self.kn)
        self.col_indices = range(self.kn, width - self.kn)

        self._cross_nodes_yx = []
        self._end_nodes_yx = []

        self.nodes = self._extract()

    def _extract(self) -> NodeContainer:
        for row in self.row_indices:
            for col in self.col_indices:
                if self.img[row, col] == 1:  # volatile
                    self._extract_from_rc(row, col)

        return NodeContainer(self._cross_nodes_yx, self._end_nodes_yx, [])

    def _extract_from_rc(self, row: int, col: int) -> None:
        point = Point(row, col)
        point.find_positive_neighbours(self.orig_img)

        num_nb8 = point.num_neighbours

        if num_nb8 == 1:  # end node
            self._add_end_node(point)

        elif num_nb8 == 2:  # straight line or bend
            n1, n2 = point.neighbours
            if n2 in n1.four_connectivity:  # 90 degree bed
                self._add_end_node(point)

        elif num_nb8 == 3:  # bifurcation
            # neighbours' 4-conn
            neighbours_4_conn = [
                nn for n in point.neighbours for nn in n.four_connectivity
            ]

            neighours_are_connected = False
            for n in point.neighbours:
                if n in neighbours_4_conn:
                    neighours_are_connected = True
                    break

            if not neighours_are_connected:
                self._neighbours_arent_connected(point)  # changes self.img

        elif num_nb8 >= 4:  # bifurcation or crossing
            dist_one_nneighbours = [
                n.find_dist_one_points(point.neighbours) for n in point.neighbours
            ]

            neighours_are_connected = False
            for n in dist_one_nneighbours:
                if len(n) >= 2:
                    neighours_are_connected = True
                    break

            if 0 not in [len(n) for n in dist_one_nneighbours]:
                neighours_are_connected = True

            if not neighours_are_connected:
                self._neighbours_arent_connected(point)

        if point.is_cross_node(self.img):
            row, col = point.row, point.col

            cross_points = [
                Point(rc)
                for rc in [
                    [row, col],
                    [row - 1, col - 1],
                    [row, col - 1],
                    [row - 1, col],
                ]
            ]

            for cp in cross_points:
                self._add_cross_node(cp)
                self._blacken(cp)

    def _neighbours_arent_connected(self, point: Point) -> None:
        self._add_cross_node(point)
        self._blacken(point)

    def _blacken(self, point: Point) -> None:
        self.img[point.row, point.col] = 0

    def _add_end_node(self, point: Point) -> None:
        self._end_nodes_yx.append([point.row, point.col])

    def _add_cross_node(self, point: Point) -> None:
        self._cross_nodes_yx.append([point.row, point.col])
