import copy
import math
import os

import cv2
import numpy as np

from tools.images import four_connectivity, generate_node_pos_img
from tools.NodeContainer import NodeContainer, flip_node_coordinates
from tools.plots import plot_landmarks_img, plot_overlay, plot_poly_graph
from tools.PolyGraph import PolyGraph


def positive_neighbours(a: int, b: int, image: np.ndarray):
    # list of pixels with value 1 in in neighbourhood of [a,b]

    nb = []
    for xx in range(a - 1, a + 2):
        for yy in range(b - 1, b + 2):
            if image[xx, yy] == 1:
                nb.append([xx, yy])

    if [a, b] in nb:
        nb.remove([a, b])

    return nb


def all_neighbours(middlepoint: list):
    # list of all pixels in neihgbourhood of [a,b]

    nb = []
    for xx in range(middlepoint[0] - 1, middlepoint[0] + 2):
        for yy in range(middlepoint[1] - 1, middlepoint[1] + 2):
            nb.append([xx, yy])

    return nb


def distance(a: list, b: list):
    # distance between point a and point b
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def clean_skeleton_img(img_skeleton: np.ndarray):
    cleaned_skeleton = img_skeleton.copy()
    binary = np.uint8(img_skeleton.copy() / 255)

    kernel = 3

    bcnodes_yx = []
    endpoints_yx = []
    n = int(np.floor(kernel / 2))

    for row in range(n, binary.shape[0] - n):
        for col in range(n, binary.shape[1] - n):
            neighbours_nn = []
            bo = []
            cross = []
            aux = 0
            if binary[row, col] == 1:
                # Anzahl der Pixel mit 1 in der neighbourhood
                # werden gezählt (inkl. Mittelpunkt)
                aux += np.sum(binary[row - n : row + n + 1, col - n : col + n + 1])
                if aux == 2:  # endpoint
                    endpoints_yx.append([row, col])
                elif aux == 3:  # endpoint bei 90° Winkel
                    neighbours_nodeall = positive_neighbours(row, col, binary)
                    conn = four_connectivity(
                        neighbours_nodeall[0][0], neighbours_nodeall[0][1]
                    )
                    if neighbours_nodeall[1] in conn:
                        endpoints_yx.append([row, col])
                elif (
                    aux == 4
                ):  # Vergabelung = 4 Pixel mit 1 -> Punkt wird gelöscht, Koordinaten werden gespeichert
                    neighbours_nodeall = positive_neighbours(row, col, binary)
                    for q in range(0, len(neighbours_nodeall)):
                        neighbours_nn.append(
                            four_connectivity(
                                neighbours_nodeall[q][0], neighbours_nodeall[q][1]
                            )
                        )
                    for p in range(0, len(neighbours_nodeall)):
                        for j in range(0, len(neighbours_nn)):
                            if neighbours_nodeall[p] in neighbours_nn[j]:
                                bo.append(True)
                            else:
                                bo.append(False)
                    if not any(bo):
                        cleaned_skeleton[row, col] = 0
                        bcnodes_yx.append([row, col])
                elif aux >= 5:  # Vergabelung oder Kreuzung
                    neighbours_nodeall = positive_neighbours(row, col, binary)
                    distone_nodes = []
                    for q in range(0, len(neighbours_nodeall)):
                        distone = []
                        for p in range(0, len(neighbours_nodeall)):
                            dist = distance(
                                neighbours_nodeall[q], neighbours_nodeall[p]
                            )
                            if dist == 1:
                                distone.append(neighbours_nodeall[p])
                        distone_nodes.append(distone)
                    numneighbours = []
                    for q in range(0, len(distone_nodes)):
                        numneighbours.append(len(distone_nodes[q]))

                        # Wenn der Abstand zwischen zwei Nachbarn des Nodes 1 beträgt,
                        # dann darf kein weiterer Nachbar des Nodes existieren, der Abstand 1 zu einem der Beiden hat
                        if len(distone_nodes[q]) >= 2:
                            bo.append(True)
                        else:
                            bo.append(False)

                    # Es muss mind einen Nachbarn des Nodes geben,
                    # der nicht direkt neben einem anderen Nachbarn des Nodes liegt
                    if 0 not in numneighbours:
                        bo.append(True)
                    if not any(bo):
                        cleaned_skeleton[row, col] = 0
                        bcnodes_yx.append([row, col])

                if row < binary.shape[0] and col < binary.shape[1]:
                    if binary[row - 1, col - 1] == 1:
                        cross.append(True)
                    if binary[row + 1, col - 2] == 1:
                        cross.append(True)
                    if binary[row, col - 1] == 1:
                        cross.append(True)
                    if binary[row - 1, col] == 1:
                        cross.append(True)
                    if binary[row - 2, col - 2] == 1:
                        cross.append(True)
                    if binary[row - 2, col + 1] == 1:
                        cross.append(True)
                    if binary[row + 1, col + 1] == 1:
                        cross.append(True)

                    if len(cross) == 7:
                        bcnodes_yx.append([row, col])
                        bcnodes_yx.append([row - 1, col - 1])
                        bcnodes_yx.append([row, col - 1])
                        bcnodes_yx.append([row - 1, col])
                        cleaned_skeleton[row, col] = 0
                        cleaned_skeleton[row - 1, col - 1] = 0
                        cleaned_skeleton[row, col - 1] = 0
                        cleaned_skeleton[row - 1, col] = 0

    return cleaned_skeleton, bcnodes_yx, endpoints_yx


def edge_extraction(skeleton: np.ndarray, nodes) -> dict:
    def flip_edge_coordinates(list_of_edges):
        return [flip_node_coordinates(edge_yx) for edge_yx in list_of_edges]

    img_binary = skeleton.copy()
    img_binary[img_binary == 255] = 1

    endpoints = nodes.end_nodes_yx + nodes.border_nodes_yx
    bcnodes = nodes.crossing_nodes_yx

    endpoints_temp = endpoints.copy()

    ese_yx = []
    edge_course_yx = []

    addpoints = []
    se = []
    n = []

    i = 0

    while len(endpoints_temp) > 0:
        addpoints = []
        se = []

        point = endpoints_temp[i]
        img_binary[point[0], point[1]] = 0

        addpoints.append(point)
        endpoints_temp.remove(point)

        reached = False

        n = get_sorted_neighbours(point, img_binary)

        while reached is False:
            bo = []
            # Nachbarn abhängig von Distanz sortieren
            if len(n) > 1:
                for k in range(len(n)):
                    if n[k] in bcnodes:
                        bo.append(True)
                        reached = True
                        point = n[k]
                        addpoints.append(point)
                    elif n[k] in endpoints_temp:
                        # print('point ', n[k], ' in endpoints_temp')
                        bo.append(True)
                        reached = True
                        point = n[k]
                        addpoints.append(point)
                        endpoints_temp.remove(point)
                        img_binary[point[0], point[1]] = 0
                    if not any(bo):
                        reached = False
                        point = n[k]
                        addpoints.append(point)
                        # print('new point = ', point)
                        img_binary[point[0], point[1]] = 0

                n = get_sorted_neighbours(point, img_binary)

            elif len(n) == 1:
                # print('len == 1')
                if n[0] in bcnodes:
                    # print('point ', n[0], ' in bcnodes')
                    bo.append(True)
                    reached = True
                    point = n[0]
                    addpoints.append(point)
                elif n[0] in endpoints_temp:
                    # print('point ', n[0], ' in endpoints_temp')
                    bo.append(True)
                    reached = True
                    point = n[0]
                    addpoints.append(point)
                    endpoints_temp.remove(point)
                    img_binary[point[0], point[1]] = 0
                if not any(bo):
                    reached = False
                    if len(n) > 0:
                        point = n[0]
                        addpoints.append(point)
                        img_binary[point[0], point[1]] = 0
                        n = get_sorted_neighbours(point, img_binary)

                    else:
                        reached = True

        l = len(addpoints)
        if addpoints[0][1] > addpoints[l - 1][1]:
            addpoints.reverse()
        elif (
            addpoints[0][1] == addpoints[l - 1][1]
            and addpoints[0][0] < addpoints[l - 1][0]
        ):
            addpoints.reverse()

        se.append(addpoints[0])
        se.append(addpoints[l - 1])
        ese_yx.append(se)
        edge_course_yx.append(addpoints)

    bcnodes_temp = bcnodes.copy()

    ese_xy = []
    edge_course_xy = []

    while len(bcnodes_temp) > 0:
        i = 0
        point1 = bcnodes_temp[i]

        n1 = positive_neighbours(point1[0], point1[1], img_binary)
        n1_original = n1.copy()
        n1_4conn = four_connectivity(point1[0], point1[1])
        img_binary[point1[0], point1[1]] = 0
        bcnodes_temp.remove(point1)
        if len(n1) > 0:
            # evtl. andere nodes aus der Nachbarschaft entfernen
            found = []
            for j in range(len(n1)):
                if n1[j] in bcnodes_temp:
                    found.append(True)
                else:
                    found.append(False)
            indices = [i for i, f in enumerate(found) if f]
            delete = []
            for j in range(len(indices)):
                delete.append(n1[indices[j]])
            for j in range(len(indices)):
                n1.remove(delete[j])
        if len(n1) > 0:
            for j in range(len(n1)):
                point = n1[j]

                # don't add the same neighbour again
                if point not in addpoints and img_binary[point[0], point[1]] == 1:
                    addpoints = []
                    se = []
                    addpoints.append(point1)  # node
                    addpoints.append(point)  # first neighbour
                    n = positive_neighbours(
                        point[0], point[1], img_binary
                    )  # neighbours of first neighbours
                    img_binary[point[0], point[1]] = 0
                    reached = False
                    dont_add = False
                else:
                    reached = True
                    dont_add = True
                while reached is False:
                    bo = []
                    for k in range(len(n)):
                        # nicht in 4conn des ursprünglichen nodes -> damit es nicht wieder zurück geht
                        if n[k] in bcnodes and n[k] not in n1_4conn:
                            bo.append(True)
                            reached = True
                            point = n[k]
                            addpoints.append(point)
                            # print(point, 'in bcnodes')
                    # print(bo)
                    if not any(bo):
                        reached = False
                        if len(n) == 0 and point1 in all_neighbours(point):
                            addpoints.append(point1)
                            # print('node is start and end')
                            reached = True
                        if len(n) == 1:
                            point = n[0]
                            addpoints.append(point)
                            # print('len(n) == 1 ', point, 'added')
                            n = positive_neighbours(point[0], point[1], img_binary)
                            img_binary[point[0], point[1]] = 0
                        elif len(n) > 1:
                            dist = []
                            for p in range(len(n)):
                                # print('test point', n[p])
                                # if n[p] not in n1:
                                if n[p] not in n1_original:
                                    dist.append([distance(point, n[p]), n[p]])
                                    addpoints.append(n[p])
                                    # print('len(n) > 1 ', n[p], 'added')
                            dist_sorted = sorted(dist, key=lambda x: x[0])
                            for p in range(len(dist_sorted)):
                                img_binary[
                                    dist_sorted[p][1][0], dist_sorted[p][1][1]
                                ] = 0
                                point = dist_sorted[p][1]
                                n = positive_neighbours(point[0], point[1], img_binary)
                        else:
                            reached = True

                if not dont_add:
                    l = len(addpoints)
                    if addpoints[0][1] > addpoints[l - 1][1]:
                        addpoints.reverse()
                    elif (
                        addpoints[0][1] == addpoints[l - 1][1]
                        and addpoints[0][0] <= addpoints[l - 1][0]
                    ):
                        addpoints.reverse()
                    se.append(addpoints[0])
                    se.append(addpoints[l - 1])
                    ese_yx.append(se)
                    edge_course_yx.append(addpoints)

        ese_xy = flip_edge_coordinates(ese_yx)
        edge_course_xy = flip_edge_coordinates(edge_course_yx)

    return {"ese": ese_xy, "path": edge_course_xy}


def get_sorted_neighbours(point, img):
    """Nachbarn abhängig von Distanz sortieren."""

    n = positive_neighbours(point[0], point[1], img)

    if len(n) > 1:
        dist = [[distance(point, neighb), neighb] for neighb in n]
        dist_sorted = sorted(dist, key=lambda x: x[0])
        n = [neighb for _, neighb in dist_sorted]

    return n


def helper_polyfit(nodes, edges: dict):
    helperedges = copy.deepcopy(edges["path"])
    ese_helperedges = copy.deepcopy(edges["ese"])

    helpernodescoor = nodes.all_nodes_xy

    # order coordinates_global -> is there a circle or any other critical structure
    check_again = [True] * len(helperedges)
    len_begin = 0
    while len(check_again) > 0:
        len_check = len(check_again)
        len_end = len_begin + len_check
        check_again = []
        for i in range(len_begin, len_end):
            edge_se = ese_helperedges[i]
            edge_start, edge_end = edge_se

            # edge is a circle
            if edge_start == edge_end:
                if len(helperedges[i]) == 1:
                    # del helperedges[i]
                    # del ese_helperedges[i]
                    # can't delete because this will break the indexing
                    continue

                # same start and end (too short)
                if len(helperedges[i]) < 6:
                    del helperedges[i][-1]
                    ese_helperedges[i][1] = helperedges[i][-1]
                    helpernodescoor.append(helperedges[i][-1])
                    continue

                # edge is a circle
                if len(helperedges[i]) >= 6:
                    edge_xy_mid = split_edge(i, ese_helperedges, helperedges)
                    helpernodescoor.append(edge_xy_mid)

                    check_again.append(True)

            # occurences of edge_se within ese_helperedges
            indices = [
                j for j, points in enumerate(ese_helperedges) if points == edge_se
            ]

            # parallel edges (if any; loop doesn't execute if len(indices) == 1)
            for j in range(1, len(indices)):
                len_edge1 = len(helperedges[i])
                len_edge2 = len(helperedges[indices[j]])

                # set helperindex to the index of the longer edge
                helperindex = i if len_edge1 > len_edge2 else indices[j]

                if len(helperedges[helperindex]) > 10:
                    edge_xy_mid = split_edge(helperindex, ese_helperedges, helperedges)
                    helpernodescoor.append(edge_xy_mid)

                    check_again.append(True)

        len_begin = len_end

    return {"path": helperedges, "ese": ese_helperedges}, helpernodescoor


def helper_structural_graph(nodes, edges: dict):
    helperedges = copy.deepcopy(edges["path"])
    ese_helperedges = copy.deepcopy(edges["ese"])

    new_helpers = []

    # order coordinates_global -> is there a circle or any other critical structure
    check_again = [True] * len(helperedges)
    len_begin = 0

    while len(check_again) > 0:
        len_check = len(check_again)
        len_end = len_begin + len_check
        check_again = []

        for i in range(len_begin, len_end):
            edge_se = ese_helperedges[i]
            edge_start, edge_end = edge_se

            # edge is a circle
            if edge_start == edge_end:
                if len(helperedges[i]) >= 6:
                    edge_xy_mid = split_edge(i, ese_helperedges, helperedges)

                    nodes.add_helper_node(edge_xy_mid)
                    new_helpers.append(edge_xy_mid)

                    check_again.append(True)

            # occurences of edge_se within ese_helperedges
            indices = [j for j, point in enumerate(ese_helperedges) if point == edge_se]

            # parallel edges (if any; loop doesn't execute if len(indices) == 1)
            for j in range(1, len(indices)):
                len_edge1 = len(helperedges[i])
                len_edge2 = len(helperedges[indices[j]])

                # set helperindex to the index of the longer edge
                helperindex = i if len_edge1 > len_edge2 else indices[j]

                if len(helperedges[helperindex]) > 10:
                    edge_xy_mid = split_edge(helperindex, ese_helperedges, helperedges)

                    nodes.add_helper_node(edge_xy_mid)
                    new_helpers.append(edge_xy_mid)

                    check_again.append(True)

        len_begin = len_end

    return ese_helperedges, new_helpers


def split_edge(i, edges_se, edges_path) -> list:
    """
    Splits the i-th edge into half.

    :param i: edge index
    :param edges_se:  start and end coordinates of all edges
    :param edges_path: path coordinates of all edges
    """
    assert len(edges_se[0]) == 2

    edge_xy = edges_path[i].copy()
    idx_mid = int(np.ceil(len(edge_xy) / 2))

    edge_xy_mid = edge_xy[idx_mid]
    edge_xy_end = edge_xy[-1]

    # set new endpoint of current edge
    edges_se[i][1] = edge_xy_mid

    # split edge_xy into two:
    # halve the current edge -- set endpoint to midpoint
    # edge_end = edge_xy_mid

    # append new start and end point
    # make a new edge between the midpoint and the old endpoint
    if edge_xy_mid[0] < edge_xy_end[0]:
        edges_se.insert(i + 1, [edge_xy_mid, edge_xy_end])
    else:
        edges_se.insert(i + 1, [edge_xy_end, edge_xy_mid])

    # add the new half edge
    new_half_edge = copy.deepcopy(edge_xy[idx_mid:])
    new_half_edge.reverse()
    edges_path.insert(i + 1, new_half_edge)

    # shorted the old edge
    num_to_delete = len(new_half_edge) - 1
    edges_path[9] = edges_path[9][:-num_to_delete]

    return edge_xy_mid


def polyfit_visualize(helper_edges: dict):
    visual_degree = 5
    point_density = 2

    edges = helper_edges["path"].copy()
    ese = helper_edges["ese"].copy()

    polyfit_coor_rotated = []
    polyfit_coor_local = []
    polyfit_coor_global = []
    polyfit_coeff_visual = []
    coordinates_local = []
    coordinates_rotated = []
    polyfit_points = []

    for i in range(len(edges)):
        # global
        edge = edges[i]
        edge_se = ese[i]
        origin_global, _ = edge_se
        xo_global, yo_global = origin_global

        # local
        edge_local = get_local_edge_coords(edge, origin_global)
        coordinates_local.append(edge_local)

        # rotated
        x_rotated, y_rotated, rot_params = get_rotated_coords(edge_se, edge_local)
        c, s = rot_params
        coordinates_rotated.append([z for z in zip(x_rotated, y_rotated)])

        polyfit_points_temp = []

        one_pixel_edge = len(x_rotated) <= 1
        if one_pixel_edge:
            p = np.zeros((visual_degree + 1,))
        else:
            p = np.polyfit(x_rotated, y_rotated, visual_degree)

        y_poly_rotated = []
        x_poly_rotated = x_rotated.copy()
        polycoor_rotated = []
        polycoor_local = []
        polycoor_global = []

        for j in range(len(edge)):
            px = x_poly_rotated[j]
            py = 0

            # polynom for visualization
            for d in range(visual_degree):
                py = py + p[d] * px ** (visual_degree - d)

            py = py + p[visual_degree]
            py = round(py, 2)

            y_poly_rotated.append(py)
            polycoor_rotated.append([px, py])

            a = int(round(px * c - py * s, 0))
            b = int(round(px * s + py * c, 0))

            polycoor_local.append([a, b])
            polycoor_global.append([a + xo_global, -b + yo_global])

        if len(polycoor_global) > point_density:
            for j in range(0, len(polycoor_global), point_density):
                polyfit_points_temp.append(polycoor_global[j])

        polyfit_points.append(polyfit_points_temp)

        polyfit_coeff_visual.append([p])

        polyfit_coor_global.append(polycoor_global)
        polyfit_coor_local.append(polycoor_local)
        polyfit_coor_rotated.append(polycoor_rotated)

    polyfit_coordinates = [
        polyfit_coor_global,
        polyfit_coor_local,
        polyfit_coor_rotated,
    ]
    # edge_coordinates = [edges, coordinates_local, coordinates_rotated]

    return polyfit_coordinates


def polyfit_training(helper_edges: dict) -> dict:
    cubic_thresh = 10  # deg3 > 10, otherwise only deg2 coefficients for training

    edges = helper_edges["path"].copy()
    ese = helper_edges["ese"].copy()
    training_parameters = {"deg3": [], "deg2": [], "length": []}

    for i, edge in enumerate(edges):
        # global
        edge_se = ese[i]
        origin_global, _ = edge_se

        # local
        edge_local = get_local_edge_coords(edge, origin_global)

        # rotated
        x_rotated, y_rotated, _ = get_rotated_coords(edge_se, edge_local)

        one_pixel_edge = len(edge) <= 1
        if one_pixel_edge:
            deg_coeffs = [0, 0]
        else:
            m = max(x_rotated)
            x_rotated_norm = [xr / m for xr in x_rotated]
            p_norm_deg3 = np.polyfit(x_rotated_norm, y_rotated, 3)

            is_cubic = abs(p_norm_deg3[0]) > cubic_thresh
            deg_norm = 3 if is_cubic else 2

            p_norm = np.polyfit(x_rotated_norm, y_rotated, deg_norm)

            deg_coeffs = [p_norm[0], p_norm[1]] if is_cubic else [0, p_norm[0]]

        training_parameters["deg3"].append(deg_coeffs[0])
        training_parameters["deg2"].append(deg_coeffs[1])
        training_parameters["length"].append(len(edge))

    return training_parameters


def get_rotated_coords(edge_se, coursecoor_local):
    start_xy, end_xy = edge_se
    xo_global, yo_global = start_xy
    xe_global, ye_global = end_xy

    xo_local, yo_local = 0, 0
    xe_local, ye_local = xe_global - xo_global, -(ye_global - yo_global)

    dx = xe_local - xo_local
    dy = ye_local - yo_local

    # avoid NaN errors
    ll = np.sqrt(dx * dx + dy * dy)

    s = 0 if ll == 0 else dy / ll
    c = 0 if ll == 0 else dx / ll

    x_rotated = [int(round(xl * c + yl * s, 0)) for xl, yl in coursecoor_local]
    y_rotated = [int(round(-xl * s + yl * c, 0)) for xl, yl in coursecoor_local]

    return x_rotated, y_rotated, (c, s)


def get_local_edge_coords(edge_global, start_xy):
    xo_global, yo_global = start_xy
    return [[x - xo_global, -(y - yo_global)] for x, y in edge_global]


def generate_graph(
    ese_helper_edges, nodes, training_parameters: dict, save: bool, graph_fp: str
):
    graph = PolyGraph()
    graph.add_nodes(nodes)
    graph.add_edges(ese_helper_edges, training_parameters, nodes)

    if save:
        graph.save(graph_fp)

    return graph


def extract_graphs(conf, skip_existing):
    """Starting with the thresholded images, performs the operations
    skeletonise, node extraction, edge extraction"""

    for fp in conf.skeletonised_image_files:
        cropped_fp = fp.replace("skeleton", "cropped")
        landmarks_fp = fp.replace("skeleton", "landmarks")
        poly_fp = fp.replace("skeleton", "poly_graph")
        overlay_fp = fp.replace("skeleton", "overlay")

        node_pos_img_fp = fp.replace("skeleton", "node_positions")
        node_pos_vec_fp = (
            os.path.splitext(fp.replace("skeleton", "node_positions"))[0] + ".npy"
        )
        adj_matr_fp = os.path.splitext(fp.replace("skeleton", "adj_matr"))[0] + ".npy"

        img_cropped = cv2.imread(cropped_fp, cv2.IMREAD_COLOR)
        img_skel = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)

        # exit if no raw image found
        if img_skel is None:
            print(f"No skeletonised image found.")
            raise Exception

        # skip already processed frames
        if os.path.isfile(overlay_fp) and skip_existing:
            continue

        # graph
        graph, nodes, cleaned_skel, pf_coords, helper_nodes = extract_graph(
            img_skel, fp, conf.graph_save
        )

        # landmarks
        plot_landmarks_img(
            nodes,
            helper_nodes["sg"],
            cleaned_skel,
            conf.lm_plot,
            conf.lm_save,
            landmarks_fp,
        )

        # numpy files
        if conf.node_pos_save:
            graph.save_positions(node_pos_vec_fp)

        if conf.node_pos_img_save:
            node_pos_img = generate_node_pos_img(graph, conf.img_length)
            cv2.imwrite(node_pos_img_fp, node_pos_img)

        if conf.adj_matr_save:
            graph.save_extended_adj_matrix(adj_matr_fp)

        # polynomial graph and overlay
        visualise_poly = conf.poly_plot or conf.poly_save
        visualise_overlay = conf.overlay_plot or conf.overlay_save

        if visualise_poly or visualise_overlay:
            edge_width = 2

            if visualise_poly:
                node_size = 10
                plot_poly_graph(
                    conf.img_length,
                    helper_nodes["pf"],
                    pf_coords,
                    conf.poly_plot,
                    conf.poly_save,
                    node_size,
                    edge_width,
                    poly_fp,
                )

            if visualise_overlay:
                node_size = 7
                plot_overlay(
                    img_cropped,
                    helper_nodes["pf"],
                    pf_coords,
                    conf.overlay_plot,
                    conf.overlay_save,
                    node_size,
                    edge_width,
                    overlay_fp,
                )


def extract_graph(img_preproc, skel_fp, graph_save=False):
    graph_fp = os.path.splitext(skel_fp.replace("skeleton", "graphs"))[0] + ".json"

    nodes, edges, cleaned_skeleton = extract_nodes_and_edges(img_preproc)
    helper_pf_edges, helper_pf_nodes = helper_polyfit(nodes, edges)
    helper_sg_edges, helper_sg_nodes = helper_structural_graph(nodes, edges)

    polyfit_params = polyfit_training(helper_pf_edges)
    pf_coords = polyfit_visualize(helper_pf_edges)

    graph = generate_graph(helper_sg_edges, nodes, polyfit_params, graph_save, graph_fp)

    return (
        graph,
        nodes,
        cleaned_skeleton,
        pf_coords,
        {"pf": helper_pf_nodes, "sg": helper_sg_nodes},
    )


def extract_nodes_and_edges(img_preproc: np.ndarray):
    """
    Extracts the nodes and edges from the skeletonised image.

    :param img_preproc:
    :return:
    """
    # clean the skeletonised image
    cleaned_skeleton, bcnodes_yx, endpoints_yx = clean_skeleton_img(img_preproc)

    # extract nodes and edges
    nodes = NodeContainer(bcnodes_yx, endpoints_yx, [])
    edges = edge_extraction(img_preproc, nodes)

    return nodes, edges, cleaned_skeleton
