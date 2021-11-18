import json

import numpy as np
import networkx as nx


class PolyGraph(nx.Graph):
    """
    Graph with polynomial edge attributes.
    """

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            data_dict = json.load(f)
            graph = nx.node_link_graph(data_dict)
        return cls(graph)

    def add_nodes(self, nodes):
        for i, xy in enumerate(nodes.all_nodes_xy):
            self.add_node(i, pos=tuple(xy), type=nodes.node_types[i])

    def add_edges(self, ese_xy, attributes: dict, nodes):
        all_nodes = nodes.all_nodes_xy

        for i, edge_se in enumerate(ese_xy):
            start, end = edge_se

            if start in all_nodes and end in all_nodes:
                startidx = all_nodes.index(start)
                endidx = all_nodes.index(end)

                self.add_edge(startidx, endidx,
                              label=i,
                              length=attributes['length'][i],
                              deg3=attributes['deg3'][i],
                              deg2=attributes['deg2'][i])

    def save(self, filepath: str):
        graph_data = nx.node_link_data(self)
        with open(filepath, 'w') as f:
            json.dump(graph_data, f)

    def save_positions(self, filepath: str):
        np.save(filepath, np.array(self.positions))

    def save_extended_adj_matrix(self, filepath: str):
        np.save(filepath, self.extended_adj_matrix)

    @property
    def positions(self):
        return list(nx.get_node_attributes(self, 'pos').values())

    @property
    def node_types(self):
        return list(nx.get_node_attributes(self, 'type').values())

    @property
    def extended_adj_matrix(self, nodelist=None):
        adj_matrix = nx.convert_matrix.to_numpy_array(self, nodelist=nodelist)
        positions_vector = np.array(self.positions)

        num_nodes = len(self)
        extended_adj_matrix = np.zeros((4, num_nodes, 2 + num_nodes))

        length_matrix = nx.attr_matrix(self, edge_attr="length")[0]
        coeff3_matrix = nx.attr_matrix(self, edge_attr="deg3")[0]
        coeff2_matrix = nx.attr_matrix(self, edge_attr="deg2")[0]

        extended_adj_matrix[0, :, :2] = positions_vector
        extended_adj_matrix[0, :, 2:] = adj_matrix

        extended_adj_matrix[1, :, 2:] = length_matrix
        extended_adj_matrix[2, :, 2:] = coeff3_matrix
        extended_adj_matrix[3, :, 2:] = coeff2_matrix

        return extended_adj_matrix
