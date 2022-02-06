from typing import List, Tuple

import cv2
import numpy as np

from tools.Point import all_neighbours, distance, four_connectivity, positive_neighbours


def node_extraction(
    cleaned_skeleton,
) -> Tuple[List[List[int]], List[List[int]], np.ndarray]:
    node_thick = 3

    skeleton = cleaned_skeleton.copy()
    skeleton_img = skeleton.copy()
    kernel = 3
    bc = []
    endpoints = []
    binary = skeleton_img.copy()
    binary[binary == 255] = 1
    result = skeleton_img.copy()
    n = int(np.floor(kernel / 2))
    for it_x in range(n, binary.shape[0] - n):
        for it_y in range(n, binary.shape[1] - n):
            neighbours_nodeall = []
            neighbours_nn = []
            bo = []
            cross = []
            aux = 0
            if binary[it_x, it_y] == 1:
                aux += np.sum(
                    binary[it_x - n : it_x + n + 1, it_y - n : it_y + n + 1]
                )  # Anzahl der Pixel mit 1 in der neighbourhood werden gezählt (inkl. Mittelpunkt)
                if aux == 2:  # endpoint
                    endpoints.append([it_x, it_y])
                if aux == 3:  # endpoint bei 90° Winkel
                    neighbours_nodeall = positive_neighbours(it_x, it_y, binary)
                    conn = four_connectivity(
                        neighbours_nodeall[0][0], neighbours_nodeall[0][1]
                    )
                    if neighbours_nodeall[1] in conn:
                        endpoints.append([it_x, it_y])
                if (
                    aux == 4
                ):  # Vergabelung = 4 Pixel mit 1 -> Punkt wird gelöscht, Koordinaten werden gespeichert
                    neighbours_nodeall = positive_neighbours(it_x, it_y, binary)
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
                        result[it_x, it_y] = 0
                        bc.append([it_x, it_y])
                elif aux >= 5:  # Vergabelung oder Kreuzung
                    neighbours_nodeall = positive_neighbours(it_x, it_y, binary)
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
                        if (
                            len(distone_nodes[q]) >= 2
                        ):  # Wenn der Abstand zwischen zwei Nachbarn des Nodes 1 beträgt,
                            bo.append(
                                True
                            )  # dann darf kein weiterer Nachbar des Nodes existieren, der Abstand 1 zu einem der Beiden hat
                        else:
                            bo.append(False)
                    if (
                        0 not in numneighbours
                    ):  # Es muss mind einen Nachbarn des Nodes geben, der nicht direkt neben einem anderen Nachbarn des Nodes liegt
                        bo.append(True)
                    if not any(bo):
                        result[it_x, it_y] = 0
                        bc.append([it_x, it_y])
                if it_x < binary.shape[0] and it_y < binary.shape[1]:
                    if binary[it_x - 1, it_y - 1] == 1:
                        cross.append(True)
                    if binary[it_x + 1, it_y - 2] == 1:
                        cross.append(True)
                    if binary[it_x, it_y - 1] == 1:
                        cross.append(True)
                    if binary[it_x - 1, it_y] == 1:
                        cross.append(True)
                    if binary[it_x - 2, it_y - 2] == 1:
                        cross.append(True)
                    if binary[it_x - 2, it_y + 1] == 1:
                        cross.append(True)
                    if binary[it_x + 1, it_y + 1] == 1:
                        cross.append(True)
                    if len(cross) == 7:
                        # print('crossing at ', [it_x, it_y])
                        bc.append([it_x, it_y])
                        bc.append([it_x - 1, it_y - 1])
                        bc.append([it_x, it_y - 1])
                        bc.append([it_x - 1, it_y])
                        result[it_x, it_y] = 0
                        result[it_x - 1, it_y - 1] = 0
                        result[it_x, it_y - 1] = 0
                        result[it_x - 1, it_y] = 0

    # plot landmarks
    bcnodes = bc.copy()
    bcnodescoor = []
    allnodes = []
    allnodescoor = []
    endpoints_coor = []
    pltimage = result.copy()
    pltimage = cv2.cvtColor(pltimage, cv2.COLOR_GRAY2RGB)
    for i in range(len(bc)):
        allnodescoor.append([bc[i][1], bc[i][0]])
        allnodes.append([bc[i][0], bc[i][1]])
        bcnodescoor.append([bc[i][1], bc[i][0]])
        cv2.circle(pltimage, (bc[i][1], bc[i][0]), 0, (255, 0, 0), node_thick)
    for i in range(len(endpoints)):
        endpoints_coor.append([endpoints[i][1], endpoints[i][0]])
        cv2.circle(
            pltimage, (endpoints[i][1], endpoints[i][0]), 0, (0, 0, 255), node_thick
        )
        allnodes.append([endpoints[i][0], endpoints[i][1]])
        allnodescoor.append([endpoints[i][1], endpoints[i][0]])

    return bcnodes, endpoints, result  # (
    #     bcnodes,
    #     bcnodescoor,
    #     endpoints,
    #     endpoints_coor,
    #     allnodes,
    #     allnodescoor,
    #     pltimage,
    # )


def edge_extraction(skeleton, endpoints, bcnodes):
    binary = skeleton.copy()
    binary[binary == 255] = 1
    endpoints_temp = endpoints.copy()

    edge_start_end = []
    edge_course = []
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
        edge_start_end.append(se)
        edge_course.append(addpoints)

    bcnodes_temp = bcnodes.copy()
    while len(bcnodes_temp) > 0:
        i = 0
        point1 = bcnodes_temp[i]
        # print('node', point1)
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
                    n = positive_neighbours(
                        point[0], point[1], binary
                    )  # neighbours of first neighbours
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
                        if (
                            n[k] in bcnodes and n[k] not in n1_4conn
                        ):  # nicht in 4conn des ursprünglichen nodes -> damit es nicht wieder zurück geht
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
                    elif (
                        addpoints[0][1] == addpoints[l - 1][1]
                        and addpoints[0][0] <= addpoints[l - 1][0]
                    ):
                        addpoints.reverse()
                    se.append(addpoints[0])
                    se.append(addpoints[l - 1])
                    edge_start_end.append(se)
                    edge_course.append(addpoints)
        coursecoor = []
        esecoor = []
        for i in range(len(edge_course)):
            coursecoor_temp = []
            esecoor_temp = []
            for j in range(len(edge_course[i])):
                coursecoor_temp.append([edge_course[i][j][1], edge_course[i][j][0]])
            for p in range(len(edge_start_end[i])):
                esecoor_temp.append([edge_start_end[i][p][1], edge_start_end[i][p][0]])
            coursecoor.append(coursecoor_temp)
            esecoor.append(esecoor_temp)

    return edge_start_end, edge_course
