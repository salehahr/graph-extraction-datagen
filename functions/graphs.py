# import networkx as nx

from functions.im2graph import node_extraction, edge_extraction

# pos = nx.get_node_attributes(graph, 'pos')


def extract_nodes_edges(img_preproc, node_size):
    bcnodes, bcnodescoor, endpoints, endpointscoor, allnodes, allnodescoor, marked_img = node_extraction(
        img_preproc, node_size)
    edge_start_end, esecoor, edge_course, coordinates_global = edge_extraction(img_preproc, endpoints,
                                                                               bcnodes)
    return allnodescoor, coordinates_global, esecoor, marked_img
