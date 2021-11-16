import json
import math
import numpy as np
import cv2
import copy
import networkx as nx
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import skeletonize


def four_connectivity(a: int, b: int):
    # list of pixels in 4-connectivity of [a,b]
    return [[a + 1, b], [a - 1, b], [a, b + 1], [a, b - 1]]


def num_in_4connectivity(a: int, b: int, image: np.ndarray):
    # how many pixel with value 255 are in 4-connectivity of [a,b]
    neighbours = four_connectivity(a, b)

    count = 0
    for nr, nc in neighbours:
        if image[nr, nc] == 255:
            count += 1

    return count


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


def preprocess(thr_image: np.ndarray, plot: bool, save: bool, directory: str):
    """
    Creates skeletonised image
    :param thr_image: thresholded image
    :param plot:
    :param save:
    :param directory:
    :return:
    """
    edgelength = 10

    # skeletonize
    img = thr_image.copy() / 255
    img = img.astype(int)
    skeleton_noisy = skeletonize(img).astype(int) * 255

    # remove too small edges
    bool_img = (skeleton_noisy.copy() / 255).astype(bool)
    labeled = morphology.label(bool_img)
    skeleton = morphology.remove_small_objects(labeled, edgelength + 1)
    skeleton[skeleton > 0] = 255
    skeleton = np.uint8(skeleton)

    remove_bug_pixels(skeleton)
    set_black_border(skeleton)

    if plot:
        fig, axes = plt.subplots(1, 2)
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])

        axes[0].imshow(thr_image, 'gray')
        axes[0].set_title('thresholded')

        axes[1].imshow(skeleton, 'gray')
        axes[1].set_title('skeletonised')

        plt.show()

    if save:
        cv2.imwrite(directory, skeleton)

    return np.uint8(skeleton)


def remove_bug_pixels(skeleton):
    # bug pixel elimination based on
    # "Preprocessing and postprocessing for skeleton-based fingerprint minutiae extraction"
    bug_pixels = []
    for x in range(1, skeleton.shape[0] - 1):
        for y in range(1, skeleton.shape[1] - 1):
            if skeleton[x, y] == 255:
                s = num_in_4connectivity(x, y, skeleton)

                if s > 2:
                    bug_pixels.append([x, y])

    for bpx, bpy in bug_pixels:
        s = num_in_4connectivity(bpx, bpy, skeleton)

        if s > 2:
            skeleton[bpx, bpy] = 0


def set_black_border(img):
    mask = np.ones(img.shape, dtype=np.int8)

    mask[:, 0] = 0
    mask[:, -1] = 0
    mask[0, :] = 0
    mask[-1, :] = 0

    img = np.uint8(np.multiply(mask, img))


def node_extraction(img_skeleton: np.ndarray):
    cleaned_skeleton = img_skeleton.copy()
    binary = img_skeleton.copy()
    binary[binary == 255] = 1

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
                aux += np.sum(binary[row - n:row + n + 1,
                              col - n:col + n + 1])
                if aux == 2:  # endpoint
                    endpoints_yx.append([row, col])
                if aux == 3:  # endpoint bei 90° Winkel
                    neighbours_nodeall = positive_neighbours(row, col, binary)
                    conn = four_connectivity(neighbours_nodeall[0][0], neighbours_nodeall[0][1])
                    if neighbours_nodeall[1] in conn:
                        endpoints_yx.append([row, col])
                if aux == 4:  # Vergabelung = 4 Pixel mit 1 -> Punkt wird gelöscht, Koordinaten werden gespeichert
                    neighbours_nodeall = positive_neighbours(row, col, binary)
                    for q in range(0, len(neighbours_nodeall)):
                        neighbours_nn.append(four_connectivity(neighbours_nodeall[q][0], neighbours_nodeall[q][1]))
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
                            dist = distance(neighbours_nodeall[q], neighbours_nodeall[p])
                            if dist == 1:
                                distone.append(neighbours_nodeall[p])
                        distone_nodes.append(distone)
                    numneighbours = []
                    for q in range(0, len(distone_nodes)):
                        numneighbours.append(len(distone_nodes[q]))

                        # Wenn der Abstand zwischen zwei Nachbarn des Nodes 1 beträgt,
                        # dann darf kein weiterer Nachbar des Nodes existieren, der Abstand 1 zu einem der Beiden hat
                        if len(distone_nodes[q]) >= 2:
                            bo.append(
                                True)
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
                    if binary[row - 1, col - 1] == 1: cross.append(True)
                    if binary[row + 1, col - 2] == 1: cross.append(True)
                    if binary[row, col - 1] == 1: cross.append(True)
                    if binary[row - 1, col] == 1: cross.append(True)
                    if binary[row - 2, col - 2] == 1: cross.append(True)
                    if binary[row - 2, col + 1] == 1: cross.append(True)
                    if binary[row + 1, col + 1] == 1: cross.append(True)
                    if len(cross) == 7:
                        # print('crossing at ', [it_x, it_y])
                        bcnodes_yx.append([row, col])
                        bcnodes_yx.append([row - 1, col - 1])
                        bcnodes_yx.append([row, col - 1])
                        bcnodes_yx.append([row - 1, col])
                        cleaned_skeleton[row, col] = 0
                        cleaned_skeleton[row - 1, col - 1] = 0
                        cleaned_skeleton[row, col - 1] = 0
                        cleaned_skeleton[row - 1, col] = 0

    # get all nodes
    allnodes_yx = bcnodes_yx + endpoints_yx
    allnodes_xy = flip_node_coordinates(allnodes_yx)

    return bcnodes_yx, endpoints_yx, allnodes_xy, cleaned_skeleton


def edge_extraction(skeleton: np.ndarray, endpoints: list, bcnodes: list):
    binary = skeleton.copy()
    binary[binary == 255] = 1

    endpoints_temp = endpoints.copy()

    ese_yx = []
    edge_course_yx = []
    i = 0

    while len(endpoints_temp) > 0:
        addpoints = []
        se = []
        point = endpoints_temp[i]
        addpoints.append(point)
        n = positive_neighbours(point[0], point[1], binary)
        binary[point[0], point[1]] = 0
        endpoints_temp.remove(point)
        reached = False

        # Nachbarn abhängig von Distanz sortieren
        if len(n) > 1:
            dist = []
            for p in range(len(n)):
                dist.append([distance(point, n[p]), n[p]])
            dist_sorted = sorted(dist, key=lambda x: x[0])
            n = []
            for p in range(len(dist_sorted)):
                n.append(dist_sorted[p][1])
        while reached is False:
            bo = []
            # Nachbarn abhängig von Distanz sortieren
            if len(n) > 1:
                # print('len > 1')
                for k in range(len(n)):
                    if n[k] in bcnodes:
                        # print('point ', n[k], ' in bcnodes')
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
                        binary[point[0], point[1]] = 0
                    if not any(bo):
                        reached = False
                        point = n[k]
                        addpoints.append(point)
                        # print('new point = ', point)
                        binary[point[0], point[1]] = 0
                n = positive_neighbours(point[0], point[1], binary)
                if len(n) > 1:
                    dist = []
                    for p in range(len(n)):
                        dist.append([distance(point, n[p]), n[p]])
                    dist_sorted = sorted(dist, key=lambda x: x[0])
                    n = []
                    for p in range(len(dist_sorted)):
                        n.append(dist_sorted[p][1])
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
                    binary[point[0], point[1]] = 0
                if not any(bo):
                    reached = False
                    if len(n) > 0:
                        point = n[0]
                        addpoints.append(point)
                        # print('newpoint = ', point)
                        binary[point[0], point[1]] = 0
                        n = positive_neighbours(point[0], point[1], binary)
                        if len(n) > 1:
                            dist = []
                            for p in range(len(n)):
                                dist.append([distance(point, n[p]), n[p]])
                            dist_sorted = sorted(dist, key=lambda x: x[0])
                            n = []
                            for p in range(len(dist_sorted)):
                                n.append(dist_sorted[p][1])
                    else:
                        reached = True
        # print('addpoints = ', addpoints)
        l = len(addpoints)
        if addpoints[0][1] > addpoints[l - 1][1]:
            addpoints.reverse()
        elif addpoints[0][1] == addpoints[l - 1][1] and addpoints[0][0] < addpoints[l - 1][0]:
            addpoints.reverse()
        # print('added points: ', addpoints)
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

        n1 = positive_neighbours(point1[0], point1[1], binary)
        n1_original = n1.copy()
        n1_4conn = four_connectivity(point1[0], point1[1])
        binary[point1[0], point1[1]] = 0
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
                if point not in addpoints and binary[point[0], point[1]] == 1:
                    # print('neighbours: ', n1, ' nextneighbour: ', point)
                    addpoints = []
                    se = []
                    addpoints.append(point1)  # node
                    addpoints.append(point)  # first neighbour
                    n = positive_neighbours(point[0], point[1], binary)  # neighbours of first neighbours
                    binary[point[0], point[1]] = 0
                    reached = False
                    dont_add = False
                else:
                    reached = True
                    dont_add = True
                while reached is False:
                    bo = []
                    # print('neighbours ', n)
                    for k in range(len(n)):
                        if n[k] in bcnodes and n[
                            k] not in n1_4conn:  # nicht in 4conn des ursprünglichen nodes -> damit es nicht wieder zurück geht
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
                            n = positive_neighbours(point[0], point[1], binary)
                            binary[point[0], point[1]] = 0
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
                                binary[dist_sorted[p][1][0], dist_sorted[p][1][1]] = 0
                                point = dist_sorted[p][1]
                                n = positive_neighbours(point[0], point[1], binary)
                        else:
                            reached = True

                # if reached: #print('reached')
                if not dont_add:
                    l = len(addpoints)
                    # print('addpoints', addpoints)
                    if addpoints[0][1] > addpoints[l - 1][1]:
                        addpoints.reverse()
                    elif addpoints[0][1] == addpoints[l - 1][1] and addpoints[0][0] <= addpoints[l - 1][0]:
                        addpoints.reverse()
                    se.append(addpoints[0])
                    se.append(addpoints[l - 1])
                    ese_yx.append(se)
                    edge_course_yx.append(addpoints)

        # flip yx coordinates to xy
        ese_xy = flip_edge_coordinates(ese_yx)
        edge_course_xy = flip_edge_coordinates(edge_course_yx)

    return ese_xy, edge_course_xy


def flip_node_coordinates(list_of_nodes_yx):
    return [[yx[1], yx[0]] for yx in list_of_nodes_yx]


def flip_edge_coordinates(list_of_edges):
    return [flip_node_coordinates(edge_yx) for edge_yx in list_of_edges]


def helpernodes_BasicGraph_for_polyfit(coordinates_global: list, esecoor: list, allnodescoor: list):
    helperedges = copy.deepcopy(coordinates_global)
    ese_helperedges = copy.deepcopy(esecoor)
    helpernodescoor = allnodescoor.copy()
    # order coordinates_global -> is there a circle or any other critical structure
    check_again = [True] * len(helperedges)
    len_begin = 0
    while len(check_again) > 0:
        len_check = len(check_again)
        len_end = len_begin + len_check
        check_again = []
        for i in range(len_begin, len_end):
            # for i in range(len(helperedges)):
            if ese_helperedges[i][0] == ese_helperedges[i][1]:
                if len(helperedges[i]) < 6:
                    # print('edge', i, 'same start and end')
                    del helperedges[i][-1]
                    ese_helperedges[i][1] = helperedges[i][-1]
                    helpernodescoor.append(helperedges[i][-1])
                else:
                    # print('edge', i, ' is a circle')
                    selected_elements = []
                    edge_xy = helperedges[i].copy()

                    idx_mid = int(np.ceil(len(edge_xy) / 2))
                    edge_xy_mid = edge_xy[idx_mid]
                    edge_xy_last = edge_xy[-1]

                    ese_helperedges[i][1] = edge_xy_mid

                    if edge_xy_mid[0] < edge_xy[-1][0]:
                        ese_helperedges.insert(i + 1, [edge_xy_mid, edge_xy_last])
                    else:
                        ese_helperedges.insert(i + 1, [edge_xy_last, edge_xy_mid])

                    helpernodescoor.append(edge_xy_mid)
                    for j in range(idx_mid, len(edge_xy)):
                        selected_elements.append(edge_xy[j])
                    selected_elements.reverse()
                    helperedges.insert(i + 1, selected_elements)
                    for j in range(idx_mid + 1, len(edge_xy)):
                        del helperedges[i][-1]
                    check_again.append(True)
            double_points = ese_helperedges[i]
            indices = [j for j, points in enumerate(ese_helperedges) if points == double_points]
            if len(indices) > 1:
                for j in range(1, len(indices)):
                    selected_elements = []
                    x = []
                    len_edge1 = len(helperedges[i])
                    len_edge2 = len(helperedges[indices[j]])
                    if len_edge1 > len_edge2:
                        helperindex = i
                    else:
                        helperindex = indices[j]
                    coursecoor_global = helperedges[helperindex].copy()
                    if len(coursecoor_global) > 10:
                        # print('edge', i, ' has a double edge')
                        index = int(np.ceil(len(coursecoor_global) / 2))
                        ese_helperedges[helperindex][1] = coursecoor_global[index]
                        if coursecoor_global[index][0] < coursecoor_global[-1][0]:
                            ese_helperedges.insert(helperindex + 1, [coursecoor_global[index], coursecoor_global[-1]])
                        else:
                            ese_helperedges.insert(helperindex + 1, [coursecoor_global[-1], coursecoor_global[index]])
                        helpernodescoor.append(coursecoor_global[index])
                        for j in range(index, len(coursecoor_global)):
                            selected_elements.append(coursecoor_global[j])
                        selected_elements.reverse()
                        helperedges.insert(helperindex + 1, selected_elements)
                        for j in range(index + 1, len(coursecoor_global)):
                            del helperedges[helperindex][-1]
                        check_again.append(True)
                    # else: print('double edge ', i, 'too short')
        len_begin = len_end

    return helperedges, ese_helperedges, helpernodescoor


def helpernodes_BasicGraph_for_structure(edge_course_xy: list,
                                         ese_xy: list,
                                         allnodes_xy: list,
                                         pltimage: np.ndarray,
                                         plot: bool,
                                         save: bool,
                                         node_thick: int,
                                         dir: str):
    helperedges = copy.deepcopy(edge_course_xy)
    ese_helperedges = copy.deepcopy(ese_xy)
    helpernodescoor = allnodes_xy.copy()

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
                    edge_xy = helperedges[i].copy()

                    idx_mid = int(np.ceil(len(edge_xy) / 2))
                    edge_xy_mid = edge_xy[idx_mid]
                    edge_xy_last = edge_xy[-1]

                    # split edge_xy into two:
                    # halve the current edge -- set endpoint to midpoint
                    edge_end = edge_xy_mid

                    # make a new edge between the midpoint and the old endpoint
                    if edge_xy_mid[0] < edge_xy_last[0]:
                        ese_helperedges.insert(i + 1, [edge_xy_mid, edge_xy_last])
                    else:
                        ese_helperedges.insert(i + 1, [edge_xy_last, edge_xy_mid])

                    helpernodescoor.append(edge_xy_mid)

                    cv2.circle(pltimage, tuple(edge_xy_mid), 0, (0, 255, 0), node_thick)

                    new_half_edge = edge_xy[idx_mid:].reverse()
                    helperedges.insert(i + 1, new_half_edge)

                    for _ in edge_xy[idx_mid + 1:]:
                        del helperedges[i][-1]

                    check_again.append(True)

            indices = [j for j, point in enumerate(ese_helperedges) if point == edge_se]

            if len(indices) > 1:
                for j in range(1, len(indices)):
                    len_edge1 = len(helperedges[i])
                    len_edge2 = len(helperedges[indices[j]])
                    if len_edge1 > len_edge2:
                        helperindex = i
                    else:
                        helperindex = indices[j]

                    coursecoor_global = helperedges[helperindex].copy()
                    if len(coursecoor_global) > 10:
                        # print('edge', i, ' has a double edge')
                        index = int(np.ceil(len(coursecoor_global) / 2))
                        ese_helperedges[helperindex][1] = coursecoor_global[index]
                        if coursecoor_global[index][0] < coursecoor_global[-1][0]:
                            ese_helperedges.insert(helperindex + 1, [coursecoor_global[index], coursecoor_global[-1]])
                        else:
                            ese_helperedges.insert(helperindex + 1, [coursecoor_global[-1], coursecoor_global[index]])

                        helpernodescoor.append(coursecoor_global[index])

                        cv2.circle(pltimage, (coursecoor_global[index][0], coursecoor_global[index][1]), 0, (0, 255, 0),
                                   node_thick)

                        selected_elements = coursecoor_global[index:].reverse()
                        helperedges.insert(helperindex + 1, selected_elements)

                        for _ in coursecoor_global[index + 1:]:
                            del helperedges[helperindex][-1]
                        check_again.append(True)
                    # else: print('double edge ', i, 'too short')
        len_begin = len_end
    if plot:
        plt.figure()
        plt.imshow(pltimage)
        plt.title('Landmarks')
        plt.xticks([]), plt.yticks([])
        plt.show()
    if save:
        cv2.imwrite(dir, pltimage)

    return ese_helperedges, helpernodescoor


def polyfit_visualize(helperedges: list, ese_helperedges: list):
    visual_degree = 5
    point_density = 2

    edges = helperedges.copy()
    ese = ese_helperedges.copy()

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

    polyfit_coordinates = [polyfit_coor_global, polyfit_coor_local, polyfit_coor_rotated]
    edge_coordinates = [edges, coordinates_local, coordinates_rotated]

    return polyfit_coeff_visual, polyfit_coordinates, edge_coordinates, polyfit_points


def polyfit_training(helperedges: list, ese_helperedges: list):
    cubic_thresh = 10  # deg3 > 10, otherwise only deg2 coefficients for training

    edges = helperedges.copy()
    ese = ese_helperedges.copy()
    training_parameters = []
    polyfit_norm_deg3 = []
    for i in range(len(edges)):
        # global
        edge = edges[i]
        edge_se = ese[i]
        origin_global, _ = edge_se

        # local
        edge_local = get_local_edge_coords(edge, origin_global)

        # rotated
        x_rotated, y_rotated, _ = get_rotated_coords(edge_se, edge_local)

        m = max(x_rotated)
        x_rotated_norm = [xr / m for xr in x_rotated]

        p_norm_deg3 = np.polyfit(x_rotated_norm, y_rotated, 3)
        polyfit_norm_deg3.append(p_norm_deg3)

        is_cubic = abs(p_norm_deg3[0]) > cubic_thresh

        deg_norm = 3 if is_cubic else 2
        p_norm = np.polyfit(x_rotated_norm, y_rotated, deg_norm)

        deg_coeffs = [p_norm[0], p_norm[1]] if is_cubic else [0, p_norm[0]]
        d = len(edge)

        training_parameters.append([deg_coeffs, d])

    return training_parameters


def get_rotated_coords(edge_se, coursecoor_local):
    start_xy, end_xy = edge_se
    xo_global, yo_global = start_xy
    xe_global, ye_global = end_xy

    xo_local, yo_local = 0, 0
    xe_local, ye_local = xe_global - xo_global, -(ye_global - yo_global)

    dx = xe_local - xo_local
    dy = ye_local - yo_local
    ll = np.sqrt(dx * dx + dy * dy)
    s = dy / ll
    c = dx / ll

    x_rotated = [int(round(xl * c + yl * s, 0)) for xl, yl in coursecoor_local]
    y_rotated = [int(round(-xl * s + yl * c, 0)) for xl, yl in coursecoor_local]

    return x_rotated, y_rotated, (c, s)


def get_local_edge_coords(edge_global, start_xy):
    xo_global, yo_global = start_xy
    return [[x - xo_global, -(y - yo_global)] for x, y in edge_global]


def graph_extraction(edge_course_xy,
                     ese_xy,
                     allnodes_xy,
                     marked_img,
                     do_plot_lm,
                     do_save_lm,
                     node_size,
                     landmarks_fp,
                     training_parameters,
                     do_save_graph,
                     graph_fp
                     ):
    deg3 = [item[0][0] for item in training_parameters]
    deg2 = [item[0][1] for item in training_parameters]
    edge_length = [item[1] for item in training_parameters]

    ese_helper_edges, helper_xy = helpernodes_BasicGraph_for_structure(
        edge_course_xy, ese_xy, allnodes_xy, marked_img,
        do_plot_lm, do_save_lm,
        node_size, landmarks_fp)

    graph = nx.Graph()
    helper_xy = sort_list_of_nodes(helper_xy)

    # define nodes with attribute position
    for i, xy in enumerate(helper_xy):
        graph.add_node(i, pos=tuple(xy))

    # define edges with attributes: weight
    for p, edge in enumerate(ese_helper_edges):
        start_xy, end_xy = edge

        if start_xy in helper_xy and end_xy in helper_xy:
            startidx = helper_xy.index(start_xy)
            endidx = helper_xy.index(end_xy)

            graph.add_edge(startidx, endidx, label=p,
                           length=edge_length[p],
                           deg3=deg3[p],
                           deg2=deg2[p])

    if do_save_graph:
        graph_data = nx.node_link_data(graph)
        with open(graph_fp, 'w') as f:
            json.dump(graph_data, f)

    return graph


def sort_list_of_nodes(unsorted):
    return sorted(unsorted, key=lambda x: [x[0], x[1]])


def graph_poly(original: np.ndarray,
               helpernodescoor: list, polyfit_coordinates: list,
               plot: bool, save: bool,
               node_size: int, edge_width: int, path: str):
    visual_graph = np.zeros([original.shape[0], original.shape[1], original.shape[2]], dtype=np.int8)
    for j in range(len(helpernodescoor)):
        cv2.circle(visual_graph, (helpernodescoor[j][0], helpernodescoor[j][1]), 0, (255, 255, 255), node_size)

    for j in range(len(polyfit_coordinates[0])):
        coordinates_global = polyfit_coordinates[0][j]
        for p in range(len(coordinates_global)):
            cv2.circle(visual_graph, (coordinates_global[p][0], coordinates_global[p][1]), 0, (255, 255, 255),
                       edge_width)

    if plot:
        plt.imshow(visual_graph)
        plt.show()

    if save:
        cv2.imwrite(path, visual_graph)

    return visual_graph


def plot_graph_on_img_poly(original: np.ndarray,
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
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        plt.imshow(overlay)
        plt.show()
    if save:
        cv2.imwrite(path, overlay)

    return overlay
