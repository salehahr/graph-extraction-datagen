import networkx as nx

from functions.im2graph import node_extraction, edge_extraction


def get_positions_list(graph) -> list:
    """ Returns list of coordinate tuples """
    pos_dict = nx.get_node_attributes(graph, 'pos')
    return [xy for xy in pos_dict.values()]


def extract_nodes_edges(img_preproc, node_size):
    bcnodes, _, endpoints, _, _, allnodescoor, marked_img = node_extraction(img_preproc, node_size)
    _, esecoor, _, coordinates_global = edge_extraction(img_preproc, endpoints, bcnodes)
    return allnodescoor, coordinates_global, esecoor, marked_img
