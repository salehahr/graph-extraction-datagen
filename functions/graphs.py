import networkx as nx
import numpy as np

from functions.im2graph import node_extraction, edge_extraction


def get_positions_list(graph) -> list:
    """ Returns list of coordinate tuples """
    pos_dict = nx.get_node_attributes(graph, 'pos')
    return [xy for xy in pos_dict.values()]


def get_positions_vector(graph, do_save: bool = True, filepath: str = '') -> np.ndarray:
    pos_list = get_positions_list(graph)

    positions_vector = np.zeros((len(graph), 2))
    for i, xy in enumerate(pos_list):
        positions_vector[i, :] = xy

    if do_save and filepath:
        np.save(filepath, positions_vector)

    return positions_vector


def extract_nodes_edges(img_preproc, node_size):
    bcnodes, _, endpoints, _, _, allnodescoor, marked_img = node_extraction(img_preproc, node_size)
    _, esecoor, _, coordinates_global = edge_extraction(img_preproc, endpoints, bcnodes)
    return allnodescoor, coordinates_global, esecoor, marked_img


def get_adjacency_matrix(graph, nodelist=None,
                         do_save: bool = False, filepath:str = ''):
    adj_matrix = nx.convert_matrix.to_numpy_matrix(graph, nodelist=nodelist)
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
