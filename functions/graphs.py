import networkx as nx

from functions.im2graph import node_extraction, edge_extraction


def get_position(graph):
    """ Returns dict {'node_id': (x, y), ...} """
    return nx.get_node_attributes(graph, 'pos')


def extract_nodes_edges(img_preproc, node_size):
    bcnodes, _, endpoints, _, _, allnodescoor, marked_img = node_extraction(img_preproc, node_size)
    _, esecoor, _, coordinates_global = edge_extraction(img_preproc, endpoints, bcnodes)
    return allnodescoor, coordinates_global, esecoor, marked_img
