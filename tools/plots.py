import cv2
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from tools.images import overlay_border, get_rgb

node_size: int = 3
bgr_blue = (255, 0, 0)
bgr_green = (0, 255, 0)
bgr_red = (0, 0, 255)
bgr_yellow = (0, 255, 255)
bgr_white = (255, 255, 255)


def plot_bgr_img(img, title=''):
    n_channels = img.shape[2] if len(img.shape) >= 3 else 1
    cmap = 'gray' if n_channels == 1 else None

    image = get_rgb(img)

    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)


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


def plot_landmarks_img(nodes, helper_nodes: list,
                       skeleton: np.ndarray,
                       plot: bool, save: bool,
                       filepath: str = ''):
    img = node_types_image(nodes, skeleton=skeleton)

    for xy in helper_nodes:
        cv2.circle(img, tuple(xy), int(node_size/2), bgr_green, -1)

    if plot:
        plot_bgr_img(img, 'Landmarks')
        plt.show()
    if save:
        cv2.imwrite(filepath, img)


def node_types_image(nodes,
                     image_length=None, skeleton=None):
    img = cv2.cvtColor(skeleton.copy(), cv2.COLOR_GRAY2RGB) \
        if skeleton is not None else \
        np.zeros((image_length, image_length, 3)).astype(np.float32)

    for xy in nodes.end_nodes_xy:
        cv2.circle(img, xy, node_size, bgr_red, -1)

    for xy in nodes.crossing_nodes_xy:
        cv2.circle(img, xy, node_size, bgr_blue, -1)

    for xy in nodes.border_nodes_xy:
        cv2.circle(img, xy, node_size, bgr_yellow, -1)

    return img


def plot_overlay(original: np.ndarray,
                 nodes_xy: list, polyfit_coordinates,
                 plot: bool, save: bool,
                 node_size: int, edge_thick: int, path: str):
    overlay = original.copy()

    for x, y in nodes_xy:
        cv2.circle(overlay, (x, y), 0, (67, 211, 255), node_size)

    for j in range(len(polyfit_coordinates[0])):
        coordinates_global = polyfit_coordinates[0][j]
        for p in range(len(coordinates_global)):
            cv2.circle(overlay, (coordinates_global[p][0], coordinates_global[p][1]), 0, (67, 211, 255), edge_thick)

    if plot:
        plot_bgr_img(overlay)
        plt.show()
    if save:
        cv2.imwrite(path, overlay)

    return overlay


def plot_poly_graph(img_length: int,
                    helpernodescoor: list, polyfit_coordinates: list,
                    plot: bool, save: bool,
                    node_size: int, edge_width: int, path: str):
    visual_graph = np.zeros((img_length, img_length, 3), dtype=np.int8)

    for xy in helpernodescoor:
        cv2.circle(visual_graph, tuple(xy), node_size, bgr_white, -1)

    for j in range(len(polyfit_coordinates[0])):
        coordinates_global = polyfit_coordinates[0][j]
        for xy in coordinates_global:
            cv2.circle(visual_graph, tuple(xy), 0, bgr_white, edge_width)

    if plot:
        plt.imshow(visual_graph)
        plt.show()

    if save:
        cv2.imwrite(path, visual_graph)

    return visual_graph


def plot_border_overlay(img_lm_fp):
    """
    Plot a circular border on the landmarks image.
    """
    img_lm_border = cv2.imread(img_lm_fp, cv2.IMREAD_COLOR)
    overlay_border(img_lm_border)

    plot_bgr_img(img_lm_border, title='landmarks')
    plt.show()
