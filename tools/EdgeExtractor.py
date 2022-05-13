from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from tools.Edge import Edge, flip_edge_coordinates
from tools.images import normalise
from tools.NodeType import NodeType
from tools.Point import Point, remove_cross_nodes


class EdgeExtractor:
    """
    Extracts edges from a skeletonised image given the end nodes and crossing nodes.
    Also detects nodes to be discarded, if any edges of length 1 are found.
    """

    def __init__(self, img: np.ndarray, nodes):
        self.img = normalise(img)  # volatile
        self.img_orig = normalise(img)  # unchanging

        self.end_nodes_yx: List[Point] = [
            Point(r, c, node_type=NodeType.END)
            for (r, c) in nodes.end_nodes_yx + nodes.border_nodes_yx
        ]  # volatile
        self.cross_nodes_yx: List[Point] = [
            Point(r, c, node_type=NodeType.CROSSING)
            for (r, c) in nodes.crossing_nodes_yx
        ]  # volatile
        self.all_nodes: List[Point] = (
            self.end_nodes_yx + self.cross_nodes_yx
        )  # volatile

        self.se_yx: List[List[List[int]]] = []
        self.paths_yx: List[List[List[int]]] = []

        self.discarded: List[Edge] = []
        self.bad_nodes: List[Point] = []

        self._current_point: Optional[Point] = None
        self._current_edge: Optional[Edge] = None

        self._flag_dont_save = False

        self._extract()
        self._process_discarded()

    def _extract(self) -> None:
        """Main process of edge extraction."""
        while self.end_nodes_yx:
            self.current_point = self.end_nodes_yx[0]
            self._extract_from_end_node()
            self._reset_edge()

        while self.cross_nodes_yx:
            self.current_point = self.cross_nodes_yx[0]
            self._extract_from_cross_node()
            self._reset_edge()

    def _extract_from_end_node(self) -> None:
        """Performs edge extraction on edges that begin with end nodes."""
        while self.reached is False:
            neighbours = self._current_point.sorted_neighbours
            neighbours_are_nodes = any([n.type for n in neighbours])

            for neighbour in neighbours:
                # end the edge if the neighbour is a node
                if neighbour.type is not None:
                    self._next_is_node(neighbour)
                    break

                # move along the path
                if not neighbours_are_nodes:
                    self._next_is_not_node(neighbour)

    def _extract_from_cross_node(self) -> None:
        """Performs edge extraction on edges that begin with crossing nodes."""

        # remove the crossing node from the image and the node list
        c_point = self._current_point
        self._blacken(c_point)
        self.cross_nodes_yx.remove(c_point)

        # neighbours of c_point which aren't crossing nodes
        c_neighbours = remove_cross_nodes(c_point.neighbours)

        p_neighbours = []
        for point in c_neighbours:
            # don't add the same neighbour again
            if point not in self._current_edge.points:
                # initialise edge
                self._current_edge = Edge()
                self.current_point = c_point
                self.current_point = point

                p_neighbours = self.current_point.neighbours
            else:
                continue

            while not self.reached:
                # neighbours of point which are crossing nodes
                # nicht in 4conn des ursprünglichen nodes
                # -> damit es nicht wieder zurück geht
                p_cross_neighbours = [
                    pn
                    for pn in p_neighbours
                    if pn.type == NodeType.CROSSING
                    and pn not in c_point.four_connectivity
                ]
                neighbours_are_nodes = len(p_cross_neighbours) > 0

                # # set dont_save flag if multiple cross nodes as neighbours
                # multiple_cross_neighbours = len(p_cross_neighbours) > 1
                # if multiple_cross_neighbours:
                #     self.flag_dont_save = True

                # end the edge if the neighbour is a node
                for pn in p_cross_neighbours:
                    self._next_is_node(pn)
                    break

                # # unset flag
                # if multiple_cross_neighbours:
                #     self.flag_dont_save = False

                if not neighbours_are_nodes:
                    num_neighbours = len(p_neighbours)

                    # full circle
                    if num_neighbours == 0 and c_point in point.all_neighbours:
                        self._next_is_node(c_point)

                    # move along path
                    elif num_neighbours == 1:
                        self._next_is_not_node(p_neighbours[0])
                        p_neighbours = self.current_point.neighbours

                    # lots of non-node neighbours
                    else:
                        dist = [
                            (point.distance_to(n_p), n_p)
                            for n_p in p_neighbours
                            if n_p not in c_point.neighbours
                        ]
                        sorted_p_neighbours = sorted(dist, key=lambda x: x[0])

                        [
                            self._current_edge.add(n_p)
                            for n_p in p_neighbours
                            if n_p not in c_point.neighbours
                        ]

                        for _, n_p in sorted_p_neighbours:
                            self._blacken(n_p)

                            point = n_p
                            point.find_neighbours(self.img)
                            p_neighbours = point.neighbours

    def _next_is_node(self, next_point: Point) -> None:
        """Sets the current point to the next node, which is the end of the edge."""
        self.current_point = next_point
        self.reached = True

    def _next_is_not_node(self, next_point: Point) -> None:
        """Sets the current point to the next point, which is not at the edge end."""
        self.current_point = next_point

    def _process_discarded(self) -> None:
        """From the discarded (length 1) edges, extract the nodes to be discarded."""
        for edge in self.discarded:
            start, end = edge.start_end
            self.bad_nodes.append(start if start.type == NodeType.END else end)

    def _reset_edge(self) -> None:
        self._current_edge = None

    def _save_edge(self) -> None:
        """Saves the edge if it has length > 1."""
        edge_too_short = len(self._current_edge) <= 2

        if edge_too_short:
            self.discarded.append(self._current_edge)
            return

        self.se_yx.append(self._current_edge.start_end_yx)
        self.paths_yx.append(self._current_edge.points_yx)

    def _blacken(self, point: Point) -> None:
        """
        Sets the corresponding pixel value to black in the image.
        Automatically removes end points from the list.
        Crossing nodes need more consideration, hence are not slated for removal here.
        """
        self.img[point.row, point.col] = 0
        if point.type == NodeType.END:
            self.end_nodes_yx.remove(point)

    def _plot_surroundings(self, point: Point, orig: bool = True) -> None:
        """
        Plots the 8-neighbourhood of the point.
        :param point: centre of the neighbourhood
        :param orig: whether to use original image or current [blacked out] image
        :return:
        """
        plt.figure()

        row, col = point.row, point.col

        img = self.img_orig if orig else self.img
        img_sec = img[row - 1 : row + 2, col - 1 : col + 2]

        plt.imshow(img_sec)
        plt.show()

    @property
    def current_point(self) -> Point:
        return self._current_point

    @current_point.setter
    def current_point(self, next_point: Point) -> None:
        """
        Setting the current point automatically updates the current edge and
        initialises the point's neighbours.
        Additionally, if the point is not a crossing node, the image pixel
        at the point is set to black.
        """
        # actual value setting
        self._current_point = next_point
        if self._current_edge:
            self._current_edge.current_point = next_point
        else:
            self._current_edge = Edge(next_point)
        self._current_edge.add(next_point)

        # remove node from database only if it is not a crossing node
        if next_point.type != NodeType.CROSSING:
            self._blacken(next_point)

        # initialise neighbours
        self._current_point.find_neighbours(self.img, node_list=self.all_nodes)

    @property
    def reached(self) -> bool:
        """Whether the end of the edge is reached."""
        return self._current_edge.reached

    @reached.setter
    def reached(self, value: bool) -> None:
        """
        Only set if value is true.
        If set, automatically saves the edge.
        """
        if value is True:
            self._current_edge.reached = value

            if not self.flag_dont_save:
                self._save_edge()

    @property
    def flag_dont_save(self) -> bool:
        """Flag to suspend saving."""
        return self._flag_dont_save

    @flag_dont_save.setter
    def flag_dont_save(self, dont_save: bool) -> None:
        """
        Flag to suspend saving.
        Unsetting the flag saves the edge, given that the end is reaached.
        """
        self._flag_dont_save = dont_save
        if dont_save is False and self.reached is True:
            self._save_edge()

    @property
    def se_xy(self) -> List[List[List[int]]]:
        """Start/end xy-coordinates of all edges."""
        return flip_edge_coordinates(self.se_yx)

    @property
    def paths_xy(self) -> List[List[List[int]]]:
        """Path xy-coordinates of all edges."""
        return flip_edge_coordinates(self.paths_yx)
