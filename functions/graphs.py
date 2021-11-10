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


def get_adjacency_matrix(graph, nodelist=None, do_save: bool = False, filepath:str = ''):
    adj_matrix = nx.convert_matrix.to_numpy_matrix(graph, nodelist=nodelist)
    positions_vector = get_positions_vector(graph)

    extended_adj_matrix = np.hstack([positions_vector, adj_matrix])

    if do_save and filepath:
        np.save(filepath, extended_adj_matrix)

    return extended_adj_matrix
