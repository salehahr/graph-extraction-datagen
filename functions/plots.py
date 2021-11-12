import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from functions.im2graph import flip_node_coordinates


def plot_graph_on_img(image: np.ndarray,
                      pos: np.ndarray,
                      adjacency: np.ndarray):
    img = image.copy()
    adjacency_matrix = np.uint8(adjacency.copy())
    positions = pos.copy()

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    pos_list = []

    # for pos in positions:
    #     x, y =
    #     pos_list.append([pos[0], img.shape[0] - pos[1]])

    for i in range(len(positions)):
        pos_list.append([positions[i][0], img.shape[0] - positions[i][1]])
    p = dict(enumerate(pos_list, 0))

    graph = nx.from_numpy_matrix(adjacency_matrix)
    nx.set_node_attributes(graph, p, 'pos')

    y_lim, x_lim = img.shape[:-1]
    extent = 0, x_lim, 0, y_lim

    plt.figure(frameon=False, figsize=(20, 20))
    plt.imshow(img, extent=extent, interpolation='nearest')
    nx.draw(graph, pos=p, node_size=50, edge_color='g', width=3, node_color='r')

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