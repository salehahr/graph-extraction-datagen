from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from tools.NodeContainer import NodeContainer


class PolyGraph(nx.Graph):
    """
    Graph with polynomial edge attributes.
    (not a lie detector)
    """

    def __init__(self, incoming_graph_data: Optional[nx.Graph] = None, **attr):
        super().__init__(incoming_graph_data, **attr)

    @classmethod
    def load(cls, filepath: str) -> PolyGraph:
        """
        Loads graph from json file.
        :param filepath: json file containing graph data
        :return: graph object
        """
        with open(filepath, "r") as f:
            data_dict = json.load(f)
            graph = nx.node_link_graph(data_dict)
        return cls(graph)

    def add_nodes(self, nodes: NodeContainer) -> None:
        """
        Appends nodes and their attributes to the graph.
        :param nodes: node container object
        """
        for i, xy in enumerate(nodes.all_nodes_xy):
            self.add_node(i, pos=tuple(xy), type=nodes.node_types[i])

    def add_edges(
        self,
        ese_xy: List[List[int]],
        attributes: Dict[str, List[float]],
        nodes: NodeContainer,
    ) -> None:
        """
        Appends edges and their attributes to the graph.
        :param ese_xy: list of start and end points of the edges (in xy format)
        :param attributes: dictionary containing edge length and polynomial coefficients
        :param nodes: node container object
        """
        all_nodes = nodes.all_nodes_xy

        for i, edge_se in enumerate(ese_xy):
            start, end = edge_se

            if start in all_nodes and end in all_nodes:
                startidx = all_nodes.index(start)
                endidx = all_nodes.index(end)

                self.add_edge(
                    startidx,
                    endidx,
                    label=i,
                    length=attributes["length"][i],
                )

    def save(self, filepath: str) -> None:
        """
        Saves graph to a .json file.
        :param filepath: filepath of json file
        """
        graph_data = nx.node_link_data(self)
        with open(filepath, "w") as f:
            json.dump(graph_data, f)

    def save_positions(self, filepath: str) -> None:
        """
        Saves node positions to a .npy file
        :param filepath: node positions filepath (.npy format)
        """
        np.save(filepath, np.array(self.positions))

    def save_extended_adj_matrix(self, filepath: str):
        np.save(filepath, self.extended_adj_matrix)

    @property
    def positions(self) -> List[List[int]]:
        """(x,y) node positions."""
        return list(nx.get_node_attributes(self, "pos").values())

    @property
    def node_types(self) -> List[int]:
        """List of node types."""
        return list(nx.get_node_attributes(self, "type").values())

    @property
    def extended_adj_matrix(self) -> np.ndarray:
        """Adjacency matrix with attributes."""
        num_nodes = len(self)
        extended_adj_matrix = np.zeros((4, num_nodes, 2 + num_nodes))

        extended_adj_matrix[0, :, :2] = np.array(self.positions)
        extended_adj_matrix[0, :, 2:] = nx.convert_matrix.to_numpy_array(self)

        extended_adj_matrix[1, :, 2:] = nx.attr_matrix(self, edge_attr="length")[0]

        return extended_adj_matrix


class PolyGraphDirected(PolyGraph):
    """
    Graph with directed polynomial edge attributes.
    cb * (x - ca)^2 + c3 * x^3

    Direction 1: start -> end
    Direction 2: end -> start
    """

    def __init__(self):
        super(PolyGraphDirected, self).__init__()

    def add_edges(
        self,
        ese_xy: List[List[List[int]]],
        attributes: Dict[str, List[float]],
        nodes: NodeContainer,
    ) -> None:
        """
        Appends edges and their attributes to the graph.
        :param ese_xy: list of start and end points of the edges (in xy format)
        :param attributes: dictionary containing edge length and polynomial coefficients
        :param nodes: node container object
        """
        all_nodes = nodes.all_nodes_xy

        for i, edge_se in enumerate(ese_xy):
            start, end = edge_se

            if start in all_nodes and end in all_nodes:
                startidx = all_nodes.index(start)
                endidx = all_nodes.index(end)

                self.add_edge(
                    startidx,
                    endidx,
                    label=i,
                    length=attributes["length"][i],
                    cb_dir1=attributes["cb_dir1"][i],
                    ca_dir1=attributes["ca_dir1"][i],
                    c3_dir1=attributes["c3_dir1"][i],
                    cb_dir2=attributes["cb_dir2"][i],
                    ca_dir2=attributes["ca_dir2"][i],
                    c3_dir2=attributes["c3_dir2"][i],
                )
