import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from functions.im2graph import flip_node_coordinates


def plot_graph_on_img_straight(img_skel: np.ndarray,
                               pos: list,
                               adjacency: np.ndarray) -> None:
    """
    Function for checking if the adjacency matrix matches the image
    by overlaying the graph over the skeletonised image.
    :param img_skel: skeletonised image
    :param pos: list of position coordinates of the graph nodes
    :param adjacency: adjacency matrix of the graph
    """
    img = img_skel.copy()

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_height = img.shape[0]
    pos_dict = {i: [x, img_height - y] for i, [x, y] in enumerate(pos)}

    graph = nx.from_numpy_array(adjacency)
    nx.set_node_attributes(graph, pos_dict, 'pos')

    y_lim, x_lim = img.shape[:-1]
    extent = 0, x_lim, 0, y_lim

    plt.figure(frameon=False, figsize=(20, 20))
    plt.imshow(img, extent=extent, interpolation='nearest')
    nx.draw(graph, pos=pos_dict,
            node_size=50, node_color='r',
            edge_color='g', width=7)

    plt.show()


def plot_landmarks(bcnodes_yx, endpoints_yx, node_size, result):
    img_lm = result.copy()
    img_lm = cv2.cvtColor(img_lm, cv2.COLOR_GRAY2RGB)

    bcnodes_xy = flip_node_coordinates(bcnodes_yx)
    for xy in bcnodes_xy:
        cv2.circle(img_lm, tuple(xy), 0, (255, 0, 0), node_size)

    endpoints_xy = flip_node_coordinates(endpoints_yx)
    for xy in endpoints_xy:
        cv2.circle(img_lm, tuple(xy), 0, (0, 0, 255), node_size)

    return img_lm