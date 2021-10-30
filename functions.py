import math
import numpy as np
import cv2
import copy
import networkx as nx
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import skeletonize


def get_position_vector(pos_list):
    pos_list
    pos = np.empty((len(pos_list), 2))
    for key in pos_list.keys():
        pos[key] = np.array(pos_list[key])
    return pos


def four_connectivity(a: int, b: int):
    #list of pixels in 4-connectivity of [a,b]
    nb = [[a+1, b], [a-1, b], [a, b+1], [a, b-1]]

    return nb


def num_in_4connectivity(a: int, b: int, image: np.ndarray):
    #how many pixel with value 255 are in 4-connectivity of [a,b]
    conn = four_connectivity(a, b)
    values = []
    for i in range(0, len(conn)):
        v = 0
        v = image[conn[i][0], conn[i][1]]
        if v == 255:
            values.append(1)
        else:
            values.append(v)
    surrounding = sum(values)

    return surrounding


def positive_neighbours(a: int, b:int, image: np.ndarray):
    #list of pixels with value 1 in in neighbourhood of [a,b]
    nb = []
    for xx in range(a - 1, a + 2):
        for yy in range(b - 1, b + 2):
            if image[xx, yy] == 1:
                nb.append([xx, yy])
    if [a, b] in nb:
        nb.remove([a, b])

    return nb


def all_neighbours(middlepoint: list):
    #list of all pixels in neihgbourhood of [a,b]
    nb = []
    for xx in range(middlepoint[0]-1, middlepoint[0]+2):
        for yy in range(middlepoint[1]-1, middlepoint[1]+2):
            nb.append([xx, yy])

    return nb


def distance(a: list, b: list):
    #distance between point a and point b
    d = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
    return d


def preprocess(image: np.ndarray, kernel: tuple, edgelength: int, plot: bool, save: bool, directory: str):
    img = image.copy()

    #Gaussfilter
    blur = cv2.GaussianBlur(img, kernel,0)

    #Otsu Threshold
    ret,thresholded = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    img = thresholded.copy()
    img = img/255
    img = img.astype(int)
    skeleton = skeletonize(img)
    skeleton = skeleton.astype(int)*255

    #remove too small edges
    img = skeleton.copy()
    img = img/255
    img = img.astype(bool)
    labeled = morphology.label(img)
    cleaned = morphology.remove_small_objects(labeled, edgelength+1)

    skeleton_cleaned = np.zeros(cleaned.shape)
    skeleton_cleaned[cleaned > 0] = 255
    skeleton_cleaned = np.uint8(skeleton_cleaned)

    # bug pixel elimination based on "Preprocessing and postprocessing for skeleton-based fingerprint minutiae extraction"
    bug_pixel = []
    for x in range(1, skeleton_cleaned.shape[0] - 1):
        for y in range(1, skeleton_cleaned.shape[1] - 1):
            if skeleton_cleaned[x, y] == 255:
                s = num_in_4connectivity(x, y, skeleton_cleaned)
                if s > 2:
                    bug_pixel.append([x, y])

    for i in range(0, len(bug_pixel)):
        s = num_in_4connectivity(bug_pixel[i][0], bug_pixel[i][1], skeleton_cleaned)
        if s > 2:
            skeleton_cleaned[bug_pixel[i][0], bug_pixel[i][1]] = 0

    mask = np.ones(skeleton_cleaned.shape, dtype=np.int8)
    mask[:, 0] = 0
    mask[:, mask.shape[1]-1] = 0
    mask[0, :] = 0
    mask[mask.shape[0] - 1, :] = 0
    skeleton_filtered = np.uint8(np.multiply(mask, skeleton_cleaned))

    if plot:
        plt.imshow(skeleton_filtered, 'gray')
        plt.xticks([]), plt.yticks([])
        plt.title('preprocessed')
        plt.show()
    if save:
        cv2.imwrite(directory, skeleton_filtered)

    return skeleton_filtered


def node_extraction(cleaned_skeleton: np.ndarray, node_thick: int):
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
                aux += np.sum(binary[it_x - n:it_x + n + 1,
                              it_y - n:it_y + n + 1])  # Anzahl der Pixel mit 1 in der neighbourhood werden gezählt (inkl. Mittelpunkt)
                if aux == 2:  # endpoint
                    endpoints.append([it_x, it_y])
                if aux == 3: # endpoint bei 90° Winkel
                    neighbours_nodeall = positive_neighbours(it_x, it_y, binary)
                    conn = four_connectivity(neighbours_nodeall[0][0], neighbours_nodeall[0][1])
                    if neighbours_nodeall[1] in conn:
                        endpoints.append([it_x, it_y])
                if aux == 4:  # Vergabelung = 4 Pixel mit 1 -> Punkt wird gelöscht, Koordinaten werden gespeichert
                    neighbours_nodeall = positive_neighbours(it_x, it_y, binary)
                    for q in range(0, len(neighbours_nodeall)):
                        neighbours_nn.append(four_connectivity(neighbours_nodeall[q][0], neighbours_nodeall[q][1]))
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
                            dist = distance(neighbours_nodeall[q], neighbours_nodeall[p])
                            if dist == 1:
                                distone.append(neighbours_nodeall[p])
                        distone_nodes.append(distone)
                    numneighbours = []
                    for q in range(0, len(distone_nodes)):
                        numneighbours.append(len(distone_nodes[q]))
                        if len(distone_nodes[q]) >= 2: #Wenn der Abstand zwischen zwei Nachbarn des Nodes 1 beträgt,
                            bo.append(True)            #dann darf kein weiterer Nachbar des Nodes existieren, der Abstand 1 zu einem der Beiden hat
                        else:
                            bo.append(False)
                    if 0 not in numneighbours: #Es muss mind einen Nachbarn des Nodes geben, der nicht direkt neben einem anderen Nachbarn des Nodes liegt
                        bo.append(True)
                    if not any(bo):
                        result[it_x, it_y] = 0
                        bc.append([it_x, it_y])
                if it_x < binary.shape[0] and it_y < binary.shape[1]:
                    if binary[it_x - 1, it_y - 1] == 1: cross.append(True)
                    if binary[it_x + 1, it_y - 2] == 1: cross.append(True)
                    if binary[it_x, it_y - 1] == 1: cross.append(True)
                    if binary[it_x - 1, it_y] == 1: cross.append(True)
                    if binary[it_x - 2, it_y - 2] == 1: cross.append(True)
                    if binary[it_x - 2, it_y + 1] == 1: cross.append(True)
                    if binary[it_x + 1, it_y + 1] == 1: cross.append(True)
                    if len(cross) == 7:
                        #print('crossing at ', [it_x, it_y])
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
        cv2.circle(pltimage, (endpoints[i][1], endpoints[i][0]), 0, (0, 0, 255), node_thick)
        allnodes.append([endpoints[i][0], endpoints[i][1]])
        allnodescoor.append([endpoints[i][1], endpoints[i][0]])

    return bcnodes, bcnodescoor, endpoints, endpoints_coor, allnodes, allnodescoor, pltimage


def edge_extraction(skeleton: np.ndarray, endpoints: list, bcnodes: list):
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
        #print('endpoint: ', point)
        addpoints.append(point)
        n = positive_neighbours(point[0], point[1], binary)
        #print('positive_neighbours: ', n)
        binary[point[0], point[1]] = 0
        endpoints_temp.remove(point)
        reached = False
        #Nachbarn abhängig von Distanz sortieren
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
            #Nachbarn abhängig von Distanz sortieren
            if len(n) > 1:
                #print('len > 1')
                for k in range(len(n)):
                    if n[k] in bcnodes:
                        #print('point ', n[k], ' in bcnodes')
                        bo.append(True)
                        reached = True
                        point = n[k]
                        addpoints.append(point)
                    elif n[k] in endpoints_temp:
                        #print('point ', n[k], ' in endpoints_temp')
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
                        #print('new point = ', point)
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
                #print('len == 1')
                if n[0] in bcnodes:
                    #print('point ', n[0], ' in bcnodes')
                    bo.append(True)
                    reached = True
                    point = n[0]
                    addpoints.append(point)
                elif n[0] in endpoints_temp:
                    #print('point ', n[0], ' in endpoints_temp')
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
                        #print('newpoint = ', point)
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
        #print('addpoints = ', addpoints)
        l = len(addpoints)
        if addpoints[0][1] > addpoints[l - 1][1]:
            addpoints.reverse()
        elif addpoints[0][1] == addpoints[l - 1][1] and addpoints[0][0] < addpoints[l - 1][0]:
            addpoints.reverse()
        #print('added points: ', addpoints)
        se.append(addpoints[0])
        se.append(addpoints[l - 1])
        edge_start_end.append(se)
        edge_course.append(addpoints)

    #point1 = [1072, 997]
    bcnodes_temp = bcnodes.copy()
    while len(bcnodes_temp) > 0:
        i = 0
        point1 = bcnodes_temp[i]
        #print('node', point1)
        n1 = positive_neighbours(point1[0], point1[1], binary)
        n1_original = n1.copy()
        n1_4conn = four_connectivity(point1[0], point1[1])
        binary[point1[0], point1[1]] = 0
        bcnodes_temp.remove(point1)
        if len(n1) > 0:
            #evtl. andere nodes aus der Nachbarschaft entfernen
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
                #don't add the same neighbour again
                if point not in addpoints and binary[point[0], point[1]] == 1:
                    #print('neighbours: ', n1, ' nextneighbour: ', point)
                    addpoints = []
                    se = []
                    addpoints.append(point1) #node
                    addpoints.append(point) #first neighbour
                    n = positive_neighbours(point[0], point[1], binary) #neighbours of first neighbours
                    binary[point[0], point[1]] = 0
                    reached = False
                    dont_add = False
                else:
                    reached = True
                    dont_add = True
                while reached is False:
                    bo = []
                    #print('neighbours ', n)
                    for k in range(len(n)):
                        if n[k] in bcnodes and n[k] not in n1_4conn: #nicht in 4conn des ursprünglichen nodes -> damit es nicht wieder zurück geht
                            bo.append(True)
                            reached = True
                            point = n[k]
                            addpoints.append(point)
                            #print(point, 'in bcnodes')
                    #print(bo)
                    if not any(bo):
                        reached = False
                        if len(n) == 0 and point1 in all_neighbours(point):
                            addpoints.append(point1)
                            #print('node is start and end')
                            reached = True
                        if len(n) == 1:
                            point = n[0]
                            addpoints.append(point)
                            #print('len(n) == 1 ', point, 'added')
                            n = positive_neighbours(point[0], point[1], binary)
                            binary[point[0], point[1]] = 0
                        elif len(n) > 1:
                            dist = []
                            for p in range(len(n)):
                                #print('test point', n[p])
                                #if n[p] not in n1:
                                if n[p] not in n1_original:
                                    dist.append([distance(point, n[p]), n[p]])
                                    addpoints.append(n[p])
                                    #print('len(n) > 1 ', n[p], 'added')
                            dist_sorted = sorted(dist, key=lambda x:x[0])
                            for p in range(len(dist_sorted)):
                                binary[dist_sorted[p][1][0], dist_sorted[p][1][1]] = 0
                                point = dist_sorted[p][1]
                                n = positive_neighbours(point[0], point[1], binary)
                        else:
                            reached = True

                #if reached: #print('reached')
                if not dont_add:
                    l = len(addpoints)
                    #print('addpoints', addpoints)
                    if addpoints[0][1] > addpoints[l - 1][1]:
                        addpoints.reverse()
                    elif addpoints[0][1] == addpoints[l - 1][1] and addpoints[0][0] <= addpoints[l - 1][0]:
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

    return edge_start_end, esecoor, edge_course, coursecoor

def helpernodes_BasicGraph_for_polyfit(coordinates_global: list, esecoor: list, allnodescoor: list):
    helperedges = copy.deepcopy(coordinates_global)
    ese_helperedges = copy.deepcopy(esecoor)
    helpernodescoor = allnodescoor.copy()
    # order coordinates_global -> is there a circle or any other critical structure
    check_again = [True] * len(helperedges)
    len_begin = 0
    while len(check_again) > 0:
        len_check = len(check_again)
        len_end = len_begin+len_check
        check_again = []
        for i in range(len_begin, len_end):
            #for i in range(len(helperedges)):
            if ese_helperedges[i][0] == ese_helperedges[i][1]:
                if len(helperedges[i]) < 6:
                    #print('edge', i, 'same start and end')
                    del helperedges[i][-1]
                    ese_helperedges[i][1] = helperedges[i][-1]
                    helpernodescoor.append(helperedges[i][-1])
                else:
                    #print('edge', i, ' is a circle')
                    selected_elements = []
                    x = []
                    coursecoor_global = helperedges[i].copy()
                    index = int(np.ceil(len(coursecoor_global) / 2))
                    ese_helperedges[i][1] = coursecoor_global[index]
                    if coursecoor_global[index][0] < coursecoor_global[-1][0]:
                        ese_helperedges.insert(i + 1, [coursecoor_global[index], coursecoor_global[-1]])
                    else:
                        ese_helperedges.insert(i + 1, [coursecoor_global[-1], coursecoor_global[index]])
                    helpernodescoor.append(coursecoor_global[index])
                    for j in range(index, len(coursecoor_global)):
                        selected_elements.append(coursecoor_global[j])
                    selected_elements.reverse()
                    helperedges.insert(i + 1, selected_elements)
                    for j in range(index + 1, len(coursecoor_global)):
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
                    if len_edge1>len_edge2:
                        helperindex = i
                    else: helperindex = indices[j]
                    coursecoor_global = helperedges[helperindex].copy()
                    if len(coursecoor_global) > 10:
                        #print('edge', i, ' has a double edge')
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
                    #else: print('double edge ', i, 'too short')
        len_begin = len_end

    return helperedges, ese_helperedges, helpernodescoor

def helpernodes_BasicGraph_for_structure(coordinates_global: list, esecoor: list, allnodescoor: list, pltimage: np.ndarray,
                           plot: bool, save: bool, node_thick: int, dir: str):
    helperedges = copy.deepcopy(coordinates_global)
    ese_helperedges = copy.deepcopy(esecoor)
    helpernodescoor = allnodescoor.copy()
    # order coordinates_global -> is there a circle or any other critical structure
    check_again = [True] * len(helperedges)
    len_begin = 0
    while len(check_again) > 0:
        len_check = len(check_again)
        len_end = len_begin+len_check
        check_again = []
        for i in range(len_begin, len_end):
            #for i in range(len(helperedges)):
            if ese_helperedges[i][0] == ese_helperedges[i][1]:
                # if len(helperedges[i]) < 6:
                #     print('edge', i, 'same start and end')
                # else:
                if len(helperedges[i]) >= 6:
                    #print('edge', i, ' is a circle')
                    selected_elements = []
                    x = []
                    coursecoor_global = helperedges[i].copy()
                    index = int(np.ceil(len(coursecoor_global) / 2))
                    ese_helperedges[i][1] = coursecoor_global[index]
                    if coursecoor_global[index][0] < coursecoor_global[-1][0]:
                        ese_helperedges.insert(i + 1, [coursecoor_global[index], coursecoor_global[-1]])
                    else:
                        ese_helperedges.insert(i + 1, [coursecoor_global[-1], coursecoor_global[index]])
                    helpernodescoor.append(coursecoor_global[index])
                    cv2.circle(pltimage, (coursecoor_global[index][0], coursecoor_global[index][1]), 0, (0, 255, 0), node_thick)
                    for j in range(index, len(coursecoor_global)):
                        selected_elements.append(coursecoor_global[j])
                    selected_elements.reverse()
                    helperedges.insert(i + 1, selected_elements)
                    for j in range(index + 1, len(coursecoor_global)):
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
                    if len_edge1>len_edge2:
                        helperindex = i
                    else: helperindex = indices[j]
                    coursecoor_global = helperedges[helperindex].copy()
                    #coursecoor_global = helperedges[indices[j]].copy()
                    if len(coursecoor_global) > 10:
                        #print('edge', i, ' has a double edge')
                        index = int(np.ceil(len(coursecoor_global) / 2))
                        ese_helperedges[helperindex][1] = coursecoor_global[index]
                        if coursecoor_global[index][0] < coursecoor_global[-1][0]:
                            ese_helperedges.insert(helperindex + 1, [coursecoor_global[index], coursecoor_global[-1]])
                        else:
                            ese_helperedges.insert(helperindex + 1, [coursecoor_global[-1], coursecoor_global[index]])
                        helpernodescoor.append(coursecoor_global[index])
                        cv2.circle(pltimage, (coursecoor_global[index][0], coursecoor_global[index][1]), 0, (0, 255, 0),
                                   node_thick)
                        for j in range(index, len(coursecoor_global)):
                            selected_elements.append(coursecoor_global[j])
                        selected_elements.reverse()
                        helperedges.insert(helperindex + 1, selected_elements)
                        for j in range(index + 1, len(coursecoor_global)):
                            del helperedges[helperindex][-1]
                        check_again.append(True)
                    #else: print('double edge ', i, 'too short')
        len_begin = len_end
    if plot:
        plt.figure()
        plt.imshow(pltimage)
        plt.title('Landmarks')
        plt.xticks([]), plt.yticks([])
        plt.show()
    if save:
        cv2.imwrite(dir, pltimage)

    return helperedges, ese_helperedges, helpernodescoor


def polyfit_visualize(helperedges: list, ese_helperedges: list, deg: int, point_density: int):
    edges = helperedges.copy()
    ese = ese_helperedges.copy()
    polyfit_coor_rotated = []
    polyfit_coor_local = []
    polyfit_coor_global = []
    polyfit_coeff_visual = []
    coordinates_local = []
    coordinates_rotated = []
    polyfit_coordinates = []
    edge_coordinates = []
    polyfit_points = []

    for i in range(len(edges)):
        polyfit_temp = []
        coursecoor_local = []
        coursecoor_rotated = []
        polyfit_points_temp = []
        x_local = []
        y_local = []
        x_global = []
        y_global = []
        coursecoor_global = edges[i]
        origin_global = ese[i][0]
        end_global = ese[i][1]
        xo_global, yo_global = origin_global[0], origin_global[1]
        xe_global, ye_global = end_global[0], end_global[1]
        xo_local, yo_local = 0, 0
        xe_local, ye_local = xe_global - xo_global, -(ye_global - yo_global)

        x1, y1 = xo_local, yo_local
        x2, y2 = xe_local, ye_local
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx * dx + dy * dy)
        s = dy / L
        c = dx / L
        x_rotated = []
        y_rotated = []

        for j in range(0, len(coursecoor_global)):
            x_global.append(coursecoor_global[j][0])
            y_global.append(coursecoor_global[j][1])
            x_local.append(coursecoor_global[j][0] - xo_global)
            y_local.append(-(coursecoor_global[j][1] - yo_global))
            coursecoor_local.append([coursecoor_global[j][0] - xo_global, -(coursecoor_global[j][1] - yo_global)])
            a = int(round(x_local[j] * c + y_local[j] * s, 0))
            b = int(round(-x_local[j] * s + y_local[j] * c, 0))
            x_rotated.append(a)
            y_rotated.append(b)
            coursecoor_rotated.append([a, b])
        coordinates_local.append(coursecoor_local)
        coordinates_rotated.append(coursecoor_rotated)
        p = np.polyfit(x_rotated, y_rotated, deg)
        polyfit_temp.append(p)
        #print('i = ', i)

        y_poly_rotated = []
        x_poly_rotated = x_rotated.copy()
        x_poly_local = []
        y_poly_local = []
        polycoor_rotated = []
        polycoor_local = []
        polycoor_global = []
        # y_poly = p[0] * x**deg + ... + p[deg]
        for j in range(len(coursecoor_global)):
            px = x_poly_rotated[j]
            py = 0
            # polynom for visualization
            for d in range(0, deg):
                py = py + p[d] * px ** (deg - d)
            py = py + p[deg]
            py = round(py, 2)
            y_poly_rotated.append(py)
            polycoor_rotated.append([px, py])
            a = int(round(px * c - py * s, 0))
            b = int(round(px * s + py * c, 0))
            x_poly_local.append(a)
            y_poly_local.append(b)
            polycoor_local.append([a, b])
            polycoor_global.append([a + xo_global, -b + yo_global])

        if len(polycoor_global) > point_density:
            for j in range(0,len(polycoor_global),point_density):
                polyfit_points_temp.append(polycoor_global[j])

        polyfit_coeff_visual.append(polyfit_temp)
        polyfit_coor_rotated.append(polycoor_rotated)
        polyfit_coor_local.append(polycoor_local)
        polyfit_coor_global.append(polycoor_global)
        polyfit_points.append(polyfit_points_temp)

    polyfit_coordinates.append(polyfit_coor_global)
    polyfit_coordinates.append(polyfit_coor_local)
    polyfit_coordinates.append(polyfit_coor_rotated)

    edge_coordinates.append(edges)
    edge_coordinates.append(coordinates_local)
    edge_coordinates.append(coordinates_rotated)

    return polyfit_coeff_visual, polyfit_coordinates, edge_coordinates, polyfit_points


def polyfit_training(helperedges: list, ese_helperedges: list, thresh: int):
    edges = helperedges.copy()
    ese = ese_helperedges.copy()
    training_parameters = []
    polyfit_norm_deg3 = []
    for i in range(len(edges)):
        polyfit_temp_norm = []
        x_local = []
        y_local = []
        coursecoor_global = edges[i]
        origin_global = ese[i][0]
        end_global = ese[i][1]
        xo_global, yo_global = origin_global[0], origin_global[1]
        xe_global, ye_global = end_global[0], end_global[1]
        xo_local, yo_local = 0, 0
        xe_local, ye_local = xe_global - xo_global, -(ye_global - yo_global)

        x1, y1 = xo_local, yo_local
        x2, y2 = xe_local, ye_local
        dx = x2 - x1
        dy = y2 - y1
        L = np.sqrt(dx * dx + dy * dy)
        s = dy / L
        c = dx / L
        x_rotated = []
        y_rotated = []

        for j in range(0, len(coursecoor_global)):
            x_local.append(coursecoor_global[j][0] - xo_global)
            y_local.append(-(coursecoor_global[j][1] - yo_global))
            a = int(round(x_local[j] * c + y_local[j] * s, 0))
            b = int(round(-x_local[j] * s + y_local[j] * c, 0))
            x_rotated.append(a)
            y_rotated.append(b)

        m = max(x_rotated)
        x_rotated_norm = []
        for j in range(len(x_rotated)):
            x_rotated_norm.append(x_rotated[j] / m)

        p_norm_deg3 = np.polyfit(x_rotated_norm, y_rotated, 3)
        polyfit_norm_deg3.append(p_norm_deg3)
        if abs(p_norm_deg3[0]) > thresh:
            deg_norm = 3
            p_norm = np.polyfit(x_rotated_norm, y_rotated, deg_norm)
            polyfit_temp_norm.append([p_norm[0], p_norm[1]])
        else:
            deg_norm = 2
            p_norm = np.polyfit(x_rotated_norm, y_rotated, deg_norm)
            polyfit_temp_norm.append([0, p_norm[0]])

        d = len(coursecoor_global)
        # d = round(distance(origin_global, end_global), 2) #euclidean distance
        polyfit_temp_norm.append(d)

        training_parameters.append(polyfit_temp_norm)

    return training_parameters


def graph_extraction(helpernodescoor: list, ese_helperedges: list, deg3: list, deg2:list, edgelength: list,
                     label_selection: list):
    C = nx.Graph()

    #define nodes with attribute position
    for i in range(0, len(helpernodescoor)):
        x = helpernodescoor[i][0]
        #y = -(helpernodescoor[i][1])
        y = helpernodescoor[i][1]
        C.add_node(i, pos=(x, y))

    #define edges with attributes: weight
    for p in range(0, len(ese_helperedges)):
        start = ese_helperedges[p][0]
        end = ese_helperedges[p][1]
        if start in helpernodescoor and end in helpernodescoor:
            startidx = helpernodescoor.index(start)
            endidx = helpernodescoor.index(end)
            C.add_edge(startidx, endidx, label=p)
            if label_selection[0]:
                C.add_edge(startidx, endidx, label=p, length=edgelength[p])
            if label_selection[1]:
                C.add_edge(startidx, endidx, label=p, deg3=deg3[p])
            if label_selection[2]:
                C.add_edge(startidx, endidx, label=p, deg2=deg2[p])

    return C

def graph_straight(Graph: dict, node_size: int, edge_width: int, plot:bool, save:bool, dir:str):
    pos = nx.get_node_attributes(Graph, "pos")
    pos_plot = pos.copy()
    for i in range(len(pos)):
        pos_plot[i] = (pos[i][0], -pos[i][1])
    #nx.set_node_attributes(Graph, pos, "pos")
    fig = plt.figure(figsize=(20, 20))
    #plt.gca().invert_yaxis
    nx.draw_networkx_edges(Graph, pos_plot, width= edge_width, alpha=0.4)
    nx.draw_networkx_nodes(Graph, pos_plot, node_size=node_size)
    plt.axis("off")
    if save:
        plt.savefig(dir)
    if plot:
        plt.show()

    return fig


def graph_poly(original: np.ndarray, helpernodescoor: list, polyfit_coordinates: list, plot: bool, save: bool,
               node_thick: int, edge_thick: int, dir):
    visual_graph = np.zeros([original.shape[0], original.shape[1], original.shape[2]], dtype=np.int8)
    for j in range(len(helpernodescoor)):
        cv2.circle(visual_graph, (helpernodescoor[j][0], helpernodescoor[j][1]), 0, (255, 255, 255), node_thick)
    for j in range(len(polyfit_coordinates[0])):
        coordinates_global = polyfit_coordinates[0][j]
        for p in range(len(coordinates_global)):
            cv2.circle(visual_graph, (coordinates_global[p][0], coordinates_global[p][1]), 0, (255, 255, 255), edge_thick)
    if plot:
        plt.imshow(visual_graph)
        plt.show()
    if save:
        cv2.imwrite(dir, visual_graph)

    return visual_graph


def plot_graph_on_img_straight(original: np.ndarray, C: dict, node_size: int, node_color: str, edge_width: int, edge_color: str, fig1_plot: bool,
                        fig1_save: bool, dir: str):
    original_graph = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    pos = nx.get_node_attributes(C, "pos")
    pos_list = pos.copy()
    positions = get_position_vector(pos_list)
    pos_list = []
    for i in range(len(positions)):
        pos_list.append([positions[i][0], original_graph.shape[0]-positions[i][1]])
    p = dict(enumerate(pos_list, 0))

    y_lim, x_lim = original_graph.shape[:-1]
    extent = 0, x_lim, 0, y_lim

    fig = plt.figure(frameon=False, figsize=(20, 20))
    plt.imshow(original_graph, extent=extent, interpolation='nearest')
    nx.draw(C, pos=p, node_size=node_size, edge_color=edge_color, width=edge_width, node_color=node_color)

    if fig1_save:
        plt.savefig(dir)
    if fig1_plot:
        plt.show()

    return fig


def plot_graph_on_img_poly(original: np.ndarray, helpernodescoor: list, polyfit_coordinates, plot: bool, save:bool,
                           node_thick: int, edge_thick: int, dir: str):
    original_graph = original.copy()
    overlay = original_graph.copy()
    for j in range(len(helpernodescoor)):
        #cv2.circle(overlay, (helpernodescoor[j][0], helpernodescoor[j][1]), 0, (0, 255, 255), node_thick)
        cv2.circle(overlay, (helpernodescoor[j][0], helpernodescoor[j][1]), 0, (67, 211, 255), node_thick)
    for j in range(len(polyfit_coordinates[0])):
        coordinates_global = polyfit_coordinates[0][j]
        for p in range(len(coordinates_global)):
            #cv2.circle(overlay, (coordinates_global[p][0], coordinates_global[p][1]), 0, (67, 211, 255), edge_thick)
            cv2.circle(overlay, (coordinates_global[p][0], coordinates_global[p][1]), 0, (67, 211, 255), edge_thick)
    #alpha = 0.5
    #image_new = cv2.addWeighted(overlay, alpha, original_graph, alpha, 1.0)
    if plot:
        #plt.imshow(image_new)
        plt.imshow(overlay)
        plt.show()
    if save:
        #cv2.imwrite(dir, image_new)
        cv2.imwrite(dir, overlay)

    return overlay  #image_new
