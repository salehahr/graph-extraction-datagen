from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING, Dict, List, Tuple

import cv2
import numpy as np

from tools import EdgeExtractor, NodeExtractor, PolyGraphDirected
from tools.Edge import Edge, flip_edge_coordinates
from tools.images import generate_node_pos_img, normalise
from tools.plots import plot_landmarks_img, plot_overlay, plot_poly_graph
from tools.Point import (
    Point,
    all_neighbours,
    distance,
    four_connectivity,
    get_sorted_neighbours,
    positive_neighbours,
)

if TYPE_CHECKING:
    from tools.NodeContainer import NodeContainer

    NodePosList = List[List[int]]
    XYCoord = List[int]


def edge_extraction(skeleton: np.ndarray, nodes) -> Dict[str, List[List]]:
    img_binary = normalise(skeleton)

    def _blacken(point_) -> None:
        img_binary[point_.row, point_.col] = 0

    end_nodes = nodes.end_nodes_yx + nodes.border_nodes_yx
    crossing_nodes = nodes.crossing_nodes_yx

    edge_start_end_yx = []
    edge_yx = []

    end_nodes.reverse()
    while len(end_nodes) > 0:
        current_edge = Edge()

        point = Point(end_nodes.pop())
        _blacken(point)
        point.find_neighbours(img_binary)
        neighbours = point.sorted_neighbours

        current_edge.add(point)

        reached = False
        while reached is False:
            bo = []

            for neighbour in neighbours:
                if neighbour.list_rc in crossing_nodes:
                    bo.append(True)
                    reached = True
                    point = neighbour
                    current_edge.add(point)
                elif neighbour.list_rc in end_nodes:
                    bo.append(True)
                    reached = True
                    point = neighbour
                    current_edge.add(point)
                    end_nodes.remove(point.list_rc)
                    _blacken(point)

                if any(bo):
                    continue
                # move straight on
                else:
                    reached = False
                    point = neighbour
                    current_edge.add(point)
                    _blacken(point)

            point.find_neighbours(img_binary)
            neighbours = point.sorted_neighbours

        current_edge.sort()
        edge_start_end_yx.append(current_edge.start_end_yx)
        edge_yx.append(current_edge.points_yx)

    crossing_nodes.reverse()
    while len(crossing_nodes) > 0:
        c_point = Point(crossing_nodes.pop())
        _blacken(c_point)

        c_point.find_neighbours(img_binary)
        c_neighbours = [
            p for p in c_point.neighbours if p.list_rc not in crossing_nodes
        ]

        current_edge = Edge()

        for point in c_neighbours:
            if point in current_edge.points:
                continue

            current_edge = Edge()
            current_edge.add(c_point)  # node
            current_edge.add(point)  # first neighbour

            _blacken(point)
            point.find_neighbours(img_binary)

            while True:
                next_c_neighbour = False
                p_neighbours = point.neighbours
                sorted_neighbours = point.sorted_neighbours

                # no neighbours -- c_point is start and end
                if len(p_neighbours) == 0 and c_point in point.all_neighbours:
                    current_edge.add(c_point)
                    break

                # check for crossing neighbours not in 4conn
                # nicht in 4conn des ursprünglichen nodes -> damit es nicht wieder zurück geht
                for neighbour in p_neighbours:
                    neighbour.find_neighbours(img_binary)
                    if (
                        neighbour.list_rc in crossing_nodes
                        and neighbour not in c_point.four_connectivity
                    ):
                        current_edge.add(neighbour)
                        next_c_neighbour = True
                if next_c_neighbour:
                    break

                # otherwise, set new point
                if len(p_neighbours) == 1:
                    point = p_neighbours[0]
                    _blacken(point)
                    point.find_neighbours(img_binary)
                    current_edge.add(point)
                elif len(p_neighbours) > 1:
                    [
                        current_edge.add(p)
                        for p in p_neighbours
                        if p not in c_point.neighbours
                    ]
                    for neighbour in sorted_neighbours:
                        if neighbour not in c_point.neighbours:
                            _blacken(neighbour)
                            point = neighbour

            current_edge.sort()
            edge_start_end_yx.append(current_edge.start_end_yx)
            edge_yx.append(current_edge.points_yx)

    edge_start_end_xy = flip_edge_coordinates(edge_start_end_yx)
    edge_xy = flip_edge_coordinates(edge_yx)

    return {"ese": edge_start_end_xy, "path": edge_xy}


def helper_polyfit(nodes, edges: EdgeExtractor) -> Tuple[Dict[str, List], NodePosList]:
    helperedges = edges.paths_xy
    ese_helperedges = edges.se_xy

    helpernodescoor = nodes.all_nodes_xy

    # order coordinates_global -> is there a circle or any other critical structure
    check_again = [True] * len(helperedges)
    len_begin = 0
    while len(check_again) > 0:
        len_check = len(check_again)
        len_end = len_begin + len_check
        check_again = []
        for i in range(len_begin, len_end):
            edge_se_i = ese_helperedges[i]
            edge_start, edge_end = edge_se_i

            # edge is a circle
            if edge_start == edge_end:
                edge_i = helperedges[i]

                if len(edge_i) == 1:
                    # del helperedges[i]
                    # del ese_helperedges[i]
                    # can't delete because this will break the indexing
                    continue

                # same start and end (too short)
                if len(edge_i) < 6:
                    del edge_i[-1]
                    edge_se_i[-1] = edge_i[-1]
                    helpernodescoor.append(edge_i[-1])
                    continue

                # edge is a circle
                if len(helperedges[i]) >= 6:
                    edge_xy_mid = split_edge(i, ese_helperedges, helperedges)
                    helpernodescoor.append(edge_xy_mid)

                    check_again.append(True)

            # occurences of edge_se within ese_helperedges
            indices = [
                j for j, points in enumerate(ese_helperedges) if points == edge_se_i
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


def helper_structural_graph(nodes: NodeContainer, edges: EdgeExtractor):
    helperedges = edges.paths_xy
    ese_helperedges = edges.se_xy

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
    :param edges_se:  start and end coordinates of all edges (volatile)
    :param edges_path: path coordinates of all edges (volatile)
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
    edges_path[i] = edges_path[i][:-num_to_delete]

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
        x_rotated, y_rotated, rot_params = get_rotated_coords(edge_local)
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


def polyfit_training(
    helper_edges: Dict[str, List[List[XYCoord]]]
) -> Dict[str, List[float]]:
    """
    Generates edge attributes, making use of polynomial fitting.
    :param helper_edges: dictionary of edges and their start and end coordinates.
    :return: dictionary of edge attributes
    """
    cubic_thresh = 10  # deg3 > 10, otherwise only deg2 coefficients for training

    edges = helper_edges["path"].copy()
    ese = helper_edges["ese"].copy()
    training_parameters = {
        "cb_dir1": [],
        "ca_dir1": [],
        "c3_dir1": [],
        "cb_dir2": [],
        "ca_dir2": [],
        "c3_dir2": [],
        "length": [],
    }

    for edge_se, edge in zip(ese, edges):
        # global
        origin_global, _ = edge_se

        # local
        edge_local = get_local_edge_coords(edge, origin_global)

        # rotated
        x_rotated, y_rotated, _ = get_rotated_coords(edge_local)

        one_pixel_edge = len(edge) <= 1
        if one_pixel_edge:
            cb_dir1, ca_dir1, c3_dir1 = 0, 0, 0
            cb_dir2, ca_dir2, c3_dir2 = 0, 0, 0
        else:
            m = max(x_rotated)

            # 2 Richtungen je nachdem von welchem Knoten das Polynom startet
            # dir1: von edges[i][0] nach edges[i][-1]
            # dir2: von edges[i][-1] nach edges[i][0]
            x_rotated_norm_dir1 = [xr / m for xr in x_rotated]
            x_rotated_norm_dir2 = [xr / m - 1 for xr in x_rotated]

            # returns fitted coeffs for x^3, x^2, x^1, x^0
            p_norm_deg3_dir1 = np.polyfit(x_rotated_norm_dir1, y_rotated, 3)

            # polynomial fitting for
            #   equation:   cb * (x - ca)^2 + c3 * x^3
            #   coeffs:     [cb, ca, c3]
            is_cubic = abs(p_norm_deg3_dir1[0]) > cubic_thresh
            if is_cubic:
                p_norm_deg3_dir2 = np.polyfit(x_rotated_norm_dir2, y_rotated, 3)

                cb_dir1, ca_dir1, c3_dir1 = _get_poly3_coeffs(p_norm_deg3_dir1)
                cb_dir2, ca_dir2, c3_dir2 = _get_poly3_coeffs(p_norm_deg3_dir2)
            else:
                p_norm_dir1 = np.polyfit(x_rotated_norm_dir1, y_rotated, 2)
                p_norm_dir2 = np.polyfit(x_rotated_norm_dir2, y_rotated, 2)

                cb_dir1, ca_dir1, c3_dir1 = _get_poly2_coeffs(p_norm_dir1)
                cb_dir2, ca_dir2, c3_dir2 = _get_poly2_coeffs(p_norm_dir2)

        training_parameters["cb_dir1"].append(cb_dir1)
        training_parameters["ca_dir1"].append(ca_dir1)
        training_parameters["c3_dir1"].append(c3_dir1)
        training_parameters["cb_dir2"].append(cb_dir2)
        training_parameters["ca_dir2"].append(ca_dir2)
        training_parameters["c3_dir2"].append(c3_dir2)
        training_parameters["length"].append(len(edge))

    return training_parameters


def _get_poly2_coeffs(coeffs_deg2) -> Tuple[float, float, float]:
    """
    Obtains polynomial coefficients (cb, ca, c3) which parametrise an equation in the form
        cb * (x - ca)^2 + c3 * x^3

    Um auf diese Schreibweise zu kommen,
    wird die x-Position (= ca) des Maximums/Minimums bestimmt.

    Dazu wird zunächst die Gleichung ohne kubischen Anteil betrachtet.

    :param coeffs_deg2: quadratic polynomial coefficients starting with highest order
    :return: polynomial coefficients (cb, ca, c3)
    """
    coeff = np.poly1d(coeffs_deg2)

    # crit gibt die x-Position des Maximums/Minimums an
    crit = coeff.deriv().r

    if len(crit) > 0:  # Prüfen ob es ein Maximum gibt
        cb, ca = coeff[2], crit[0]
    else:
        cb, ca = (0, 0)
    c3 = 0

    return cb, ca, c3


def _get_poly3_coeffs(coeffs_deg3) -> Tuple[float, float, float]:
    """
    Obtains polynomial coefficients (cb, ca, c3) which parametrise an equation in the form
        cb * (x - ca)^2 + c3 * x^3

    Um auf diese Schreibweise zu kommen,
    wird die x-Position (= ca) des Maximums/Minimums bestimmt.

    Dazu wird zunächst die Gleichung ohne kubischen Anteil betrachtet.

    :param coeffs_deg3: cubic polynomial coefficients starting with highest order
    :return: polynomial coefficients (cb, ca, c3)
    """
    coeff = np.poly1d(coeffs_deg3[1:])

    # crit gibt die x-Position des Maximums/Minimums an
    crit = coeff.deriv().r

    cb = coeff[2]
    ca = crit[0]
    c3 = coeffs_deg3[0]

    return cb, ca, c3


def get_rotated_coords(edge_local: List[XYCoord]) -> Tuple[List, List, Tuple[int, int]]:
    """
    Transforms local edge to the rotated CS.
    :param edge_local: edge in local coordinates
    :return: (x, y) coordinates of rotated local edge and the rotation parameters
    """
    xo_local, yo_local = 0, 0
    xe_local, ye_local = edge_local[-1] if edge_local[0] == [0, 0] else edge_local[0]

    dx = xe_local - xo_local
    dy = ye_local - yo_local

    # avoid NaN errors
    ll = np.sqrt(dx * dx + dy * dy)

    s = 0 if ll == 0 else dy / ll
    c = 0 if ll == 0 else dx / ll

    x_rotated = [int(round(xl * c + yl * s, 0)) for xl, yl in edge_local]
    y_rotated = [int(round(-xl * s + yl * c, 0)) for xl, yl in edge_local]

    return x_rotated, y_rotated, (c, s)


def get_local_edge_coords(
    edge_global: List[XYCoord], start_xy: XYCoord
) -> List[XYCoord]:
    """
    Transforms an edge from global to local coordinates.
    :param edge_global: edge in global coordinates
    :param start_xy: starting coordinates of the edge
    :return:
    """
    xo_global, yo_global = start_xy
    return [[x - xo_global, -(y - yo_global)] for x, y in edge_global]


def generate_graph(
    edges: List[List[XYCoord]],
    nodes: NodeContainer,
    edge_attributes: Dict[str, List[float]],
    save: bool,
    graph_fp: str,
) -> PolyGraphDirected:
    """
    Generates graph object from the given nodes and edges.
    :param edges: list of edges
    :param nodes: node container object
    :param edge_attributes: dictionary of edge attributes
    :param save: whether to save graph to disk or not
    :param graph_fp: destination filepath (json file)
    :return:
    """
    graph = PolyGraphDirected()

    graph.add_nodes(nodes)
    graph.add_edges(edges, edge_attributes, nodes)

    if save:
        graph.save(graph_fp)

    return graph


def extract_graphs(conf, skip_existing: bool) -> None:
    """Starting with the thresholded images, performs the operations
    skeletonise, node extraction, edge extraction"""
    if conf.use_images:
        landmarkpath = (conf.filepath + "\\landmarks").replace("\\", "/")
        if not os.path.exists(landmarkpath):
            os.mkdir(landmarkpath)
        polygraphpath = (conf.filepath + "\\poly_graph").replace("\\", "/")
        if not os.path.exists(polygraphpath):
            os.mkdir(polygraphpath)
        overlaypath = (conf.filepath + "\\overlay").replace("\\", "/")
        if not os.path.exists(overlaypath):
            os.mkdir(overlaypath)
        nodepospath = (conf.filepath + "\\node_positions").replace("\\", "/")
        if not os.path.exists(nodepospath):
            os.mkdir(nodepospath)
        adjpath = (conf.filepath + "\\adj_matrix").replace("\\", "/")
        if not os.path.exists(adjpath):
            os.mkdir(adjpath)
        graphpath = (conf.filepath + "\\graphs").replace("\\", "/")
        if not os.path.exists(graphpath):
            os.mkdir(graphpath)

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
        graph, nodes, pf_coords, helper_nodes = extract_graph(
            img_skel, fp, conf.graph_save
        )

        # landmarks
        plot_landmarks_img(
            nodes,
            helper_nodes["sg"],
            img_skel,
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


def extract_graph(
    skel_img: np.ndarray, skel_fp: str, graph_save: bool = False
) -> Tuple[PolyGraphDirected, NodeContainer, List[List], Dict[str, List[XYCoord]]]:
    graph_fp = skel_fp.replace("skeleton", "graphs").replace(".png", ".json")

    nodes, edges = extract_nodes_and_edges(skel_img)
    helper_pf_edges, helper_pf_nodes = helper_polyfit(nodes, edges)
    helper_sg_edges, helper_sg_nodes = helper_structural_graph(nodes, edges)

    polyfit_params = polyfit_training(helper_pf_edges)
    polyfit_coords = polyfit_visualize(helper_pf_edges)

    graph = generate_graph(helper_sg_edges, nodes, polyfit_params, graph_save, graph_fp)

    return (
        graph,
        nodes,
        polyfit_coords,
        {"pf": helper_pf_nodes, "sg": helper_sg_nodes},
    )


def extract_nodes_and_edges(
    skel_img: np.ndarray,
) -> Tuple[NodeContainer, EdgeExtractor]:
    """
    Extracts the nodes and edges from the skeletonised image.
    """
    # clean the skeletonised image
    # extract nodes and edges
    nodes = NodeExtractor(skel_img).nodes
    edges = EdgeExtractor(skel_img, nodes)

    return nodes, edges
