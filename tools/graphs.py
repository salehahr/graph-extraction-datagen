import json

import networkx as nx
import numpy as np


def get_positions_list(graph) -> list:
    """ Returns list of coordinate tuples """
    return list(nx.get_node_attributes(graph, 'pos').values())


def get_positions_vector(graph, do_save: bool = True, filepath: str = '') -> np.ndarray:
    pos = np.array(get_positions_list(graph))

    if do_save and filepath:
        np.save(filepath, pos)

    return pos


def get_ext_adjacency_matrix(graph, nodelist=None,
                             do_save: bool = False, filepath:str = ''):
    adj_matrix = nx.convert_matrix.to_numpy_array(graph, nodelist=nodelist)
    positions_vector = get_positions_vector(graph)

    num_nodes = len(graph)
    extended_adj_matrix = np.zeros((4, num_nodes, 2 + num_nodes))
    length_matrix = nx.attr_matrix(graph, edge_attr="length")[0]
    coeff3_matrix = nx.attr_matrix(graph, edge_attr="deg3")[0]
    coeff2_matrix = nx.attr_matrix(graph, edge_attr="deg2")[0]

    extended_adj_matrix[0, :, :2] = positions_vector
    extended_adj_matrix[0, :, 2:] = adj_matrix

    extended_adj_matrix[1, :, 2:] = length_matrix
    extended_adj_matrix[2, :, 2:] = coeff3_matrix
    extended_adj_matrix[3, :, 2:] = coeff2_matrix

    if do_save and filepath:
        np.save(filepath, extended_adj_matrix)

    return extended_adj_matrix


def load_graph(filepath):
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
        graph = nx.node_link_graph(data_dict)
    return graph
