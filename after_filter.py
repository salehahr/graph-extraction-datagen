import cv2
import networkx as nx
import numpy as np
import glob
import os
import sys

from functions.images import thresholding
from functions.im2graph import preprocess, node_extraction, edge_extraction, helpernodes_BasicGraph_for_polyfit, \
    helpernodes_BasicGraph_for_structure
from functions.im2graph import polyfit_visualize, polyfit_training
from functions.im2graph import graph_extraction, graph_poly
from functions.im2graph import plot_graph_on_img_poly

from functions.images import apply_img_mask

import config

import warnings

warnings.simplefilter('ignore', np.RankWarning)

gausskernel = (5, 5)

conf = config.Config()

# apply mask
apply_img_mask(conf)

# load filtered images
for filepath_filt in conf.masked_image_files:
    filepath_orig = filepath_filt.replace('masked', 'cropped')
    filepath_thr = filepath_filt.replace('masked', 'threshed')
    filepath_pr = filepath_filt.replace('masked', 'skeleton')
    filepath_lm = filepath_filt.replace('masked', 'landmarks')
    filepath_poly = filepath_filt.replace('masked', 'poly_graph')
    filepath_overlay = filepath_filt.replace('masked', 'overlay')

    original = cv2.imread(filepath_orig, cv2.IMREAD_COLOR)
    if original is None:
        print(f'No original found.')
        sys.exit(1)

    filtered_img = cv2.imread(filepath_filt, 0)

    # thresholding
    thresholded = thresholding(filtered_img, config.thr_save, filepath_thr)

    # skeletonise
    edgelength = 10
    preprocessed_image = preprocess(thresholded, original, edgelength,
                                    config.pr_plot, config.pr_save, filepath_pr)

    # landmarks
    node_thick = 6
    bcnodes, bcnodescoor, endpoints, endpointscoor, allnodes, allnodescoor, marked_img = node_extraction(
            preprocessed_image, node_thick)

    edge_start_end, esecoor, edge_course, coordinates_global = edge_extraction(preprocessed_image, endpoints, bcnodes)

    helperedges, ese_helperedges, helpernodescoor = helpernodes_BasicGraph_for_polyfit(coordinates_global, esecoor,
            allnodescoor)

    helperedges_structure, ese_helperedges_structure, helpernodescoor_structure = helpernodes_BasicGraph_for_structure(
            coordinates_global, esecoor,allnodescoor, marked_img,  config.lm_plot, config.lm_save, node_thick, filepath_lm)

    # polynomial
    visual_degree = 5
    point_density = 2
    cubic_thresh = 10 #deg3 > 10, otherwise only deg2 coefficients for training
    polyfit_coeff_visual, polyfit_coordinates, edge_coordinates, polyfit_points = polyfit_visualize(helperedges,
            ese_helperedges, visual_degree, point_density)

    training_parameters = polyfit_training(helperedges, ese_helperedges, cubic_thresh)

    deg3 = [item[0][0] for item in training_parameters]
    deg2 = [item[0][1] for item in training_parameters]
    edgelength = [item[1] for item in training_parameters]
    edgelength_bool = True
    deg3_bool = True
    deg2_bool = True
    label_selection = [edgelength_bool, deg3_bool, deg2_bool]
    graph = graph_extraction(helpernodescoor_structure, ese_helperedges_structure, deg3, deg2, edgelength, label_selection)

    pos = nx.get_node_attributes(graph, 'pos')

    # #plot Graph, only connections
    # g_s_dir = directory + '/05_Straight_Graph/'
    # g_s_name = str(key).zfill(5) + 'straight_graph.png'
    # fig1_plot = True
    # fig1_save = False
    # node_size = 150
    # #node_size = 60
    # edge_width = 4
    # #edge_width = 2
    # fig1 = graph_straight(graph, node_size, edge_width, fig1_plot, fig1_save, g_s_dir + g_s_name)

    # #plot Graph, polynom edges
    node_thick = 10
    #node_thick = 6
    edge_thick = 2
    fig2 = graph_poly(original, helpernodescoor, polyfit_coordinates, config.poly_plot, config.poly_save, node_thick, edge_thick,
                      filepath_poly)

    # #plot graph on image, only connections
    # g_s_img_dir = directory + '/07_Straight_Graph_Original/'
    # g_s_img_name = str(key).zfill(5) + 'straight_graph_original.png'
    # #node_size = 100
    # node_size = 60
    # node_color = 'y'
    # #edge_width = 4
    # edge_width = 2
    # edge_color = 'y'
    # fig3_plot = True
    # fig3_save = False
    # fig3 = plot_graph_on_img_straight(original, graph, node_size, node_color, edge_width, edge_color, fig3_plot, fig3_save,
                                   # g_s_img_dir + g_s_img_name)

    #plot graph on image, polynom edges
    #node_thick = 6
    node_thick = 7
    edge_thick = 2
    fig4 = plot_graph_on_img_poly(original, helpernodescoor, polyfit_coordinates,
                                  config.overlay_plot, config.overlay_save,
                                  node_thick, edge_thick, filepath_overlay)



# # generate images for presentation: 00100image
# import matplotlib.pyplot as plt
# prerpocessed_image = cv2.imread('S:/06_Studienarbeit/03_CNN\generate_data\data/train_less128_2000imgs\image1/00100image.png', 0)
#
# #nodes
# node_img = np.zeros(preprocessed_image.shape)
# node_img = np.uint8(node_img)
# node_img = cv2.cvtColor(node_img, cv2.COLOR_GRAY2BGR)
# for i in range(len(allnodescoor)):
#     cv2.circle(node_img, (allnodescoor[i][0], allnodescoor[i][1]), 0, (255, 255, 255), 8)
# plt.imshow(node_img)
# plt.show()
# cv2.imwrite('S:/06_Studienarbeit/06_Abschlussvortrag\Abbildungen/Node_label.png', node_img)
#
# node_img = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
# for i in range(len(allnodescoor)):
#     cv2.circle(node_img, (allnodescoor[i][0], allnodescoor[i][1]), 0, (255, 255, 255), 8)
# plt.imshow(node_img)
# plt.show()
# cv2.imwrite('S:/06_Studienarbeit/06_Abschlussvortrag\Abbildungen/Node_on_curve.png', node_img)
#
# #helpernodes
# point_img = np.zeros(preprocessed_image.shape)
# point_img = np.uint8(point_img)
# point_img = cv2.cvtColor(point_img, cv2.COLOR_GRAY2BGR)
# cv2.circle(point_img, (230, 411), 0, (0, 255, 0), 8)
# cv2.circle(point_img, (39, 424), 0, (0, 255, 0), 8)
# plt.imshow(point_img)
# plt.show()
# cv2.imwrite('S:/06_Studienarbeit/06_Abschlussvortrag\Abbildungen/Helperpoint_label.png', point_img)
#
# hp_on_curve = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
# cv2.circle(hp_on_curve, (230, 411), 0, (0, 255, 0), 8)
# cv2.circle(hp_on_curve, (39, 424), 0, (0, 255, 0), 8)
# plt.imshow(hp_on_curve)
# plt.show()
# cv2.imwrite('S:/06_Studienarbeit/06_Abschlussvortrag\Abbildungen/Helperpoint_on_curve.png', hp_on_curve)
#
# #helpernodes evaluation
# hp_eval_img = np.zeros(preprocessed_image.shape)
# hp_eval_img = np.uint8(hp_eval_img)
# hp_eval_img = cv2.cvtColor(hp_eval_img, cv2.COLOR_GRAY2BGR)
# for j in range(len(coordinates_global[35])):
#     cv2.circle(hp_eval_img, (coordinates_global[35][j][0], coordinates_global[35][j][1]), 0, (50, 205, 50), 2)
# for j in range(len(coordinates_global[37])):
#     cv2.circle(hp_eval_img, (coordinates_global[37][j][0], coordinates_global[37][j][1]), 0, (87, 201, 0), 2)
# plt.imshow(hp_eval_img)
# plt.show()
# cv2.imwrite('S:/06_Studienarbeit/06_Abschlussvortrag\Abbildungen/Helperpoint_eval.png', hp_eval_img)
#
# eval_on_curve = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
# for j in range(len(coordinates_global[35])):
#     cv2.circle(eval_on_curve, (coordinates_global[35][j][0], coordinates_global[35][j][1]), 0, (50, 205, 50), 1)
# for j in range(len(coordinates_global[37])):
#     cv2.circle(eval_on_curve, (coordinates_global[37][j][0], coordinates_global[37][j][1]), 0, (87, 201, 0), 1)
# plt.imshow(eval_on_curve)
# plt.show()
# cv2.imwrite('S:/06_Studienarbeit/06_Abschlussvortrag\Abbildungen/Helperpoint_eval_on_curve.png', eval_on_curve)
#
#
# # plot Poly Approximation of single edge
# import matplotlib.pyplot as plt
# i = 57
# x_local = []
# y_local = []
# x_rotated = []
# y_rotated = []
# x_poly_local = []
# y_poly_local = []
# x_poly_rotated = []
# y_poly_rotated = []
# x_poly_global = []
# y_poly_global = []
# x_global = []
# y_global = []
#
# coordinates_global = edge_coordinates[0]
# coordinates_local = edge_coordinates[1]
# coordinates_rotated = edge_coordinates[2]
# polyfit_coor_global = polyfit_coordinates[0]
# polyfit_coor_local = polyfit_coordinates[1]
# polyfit_coor_rotated = polyfit_coordinates[2]
#
# for j in range(len(coordinates_local[i])):
#     x_local.append(coordinates_local[i][j][0])
#     y_local.append(coordinates_local[i][j][1])
#     x_rotated.append(coordinates_rotated[i][j][0])
#     y_rotated.append(coordinates_rotated[i][j][1])
#
# for j in range(len(coordinates_global[i])):
#     x_global.append(coordinates_global[i][j][0])
#     y_global.append(coordinates_global[i][j][1])
#
# for j in range(len(polyfit_coor_local[i])):
#     x_poly_local.append(polyfit_coor_local[i][j][0])
#     y_poly_local.append(polyfit_coor_local[i][j][1])
#     x_poly_rotated.append(polyfit_coor_rotated[i][j][0])
#     y_poly_rotated.append(polyfit_coor_rotated[i][j][1])
#     x_poly_global.append(polyfit_coor_global[i][j][0])
#     y_poly_global.append(polyfit_coor_global[i][j][1])
#
# #plt.rc('text', usetex=True)
# #plt.rc('font', family='serif')
#
# fig, ax = plt.subplots()
# #plt.plot(x_local, y_local, color='crimson', alpha=0.3, label='local coordinates')
# plt.plot(x_rotated, y_rotated, color='navy', label='Transformierte und rotierte Kante des Graphen')
# plt.plot(x_poly_rotated, y_poly_rotated, color='grey', label='Approximiertes Polynom')
# #plt.plot(x_poly_local, y_poly_local, color='blue', alpha=0.3, label='poly_local')
# ax.axis('equal')  # equal aspect ratio between x and y axis to prevent that the function would look skewed
# # show the axes at (0,0)
# ax.spines['left'].set_position('zero')  # linke Achse in den Ursprung setzen
# ax.spines['right'].set_color('none')  # obere Achse unsichtbar machen
# ax.spines['bottom'].set_position('zero')  # untere Achse in den Ursprung setzen
# ax.spines['top'].set_color('none')  # untere Achse unsichtbar machen
# ax.legend()
# plt.savefig('D:\Documents\Studium\Studienarbeit/02_Ausarbeitung/2020-08-18__ISYS_Thesis\Abbildungen\Graphenextraktion/Kantenrotation.png')
# #plt.savefig('S:/06_Studienarbeit/06_Abschlussvortrag\Abbildungen/Kantenrotation.png')
# plt.show()
#
# #Difference of Gaussian (DoG)
# import matplotlib.pyplot as plt
# import math
#
# #x = np.arange(-5, 5, 2000)
# x = np.linspace(-5, 5)
# sigma1 = 2
# sigma2 = 1
# g1 = 1 / (np.sqrt(2 * np.pi) * sigma1) * np.exp(-x ** 2 / (2 * sigma1 ** 2))
# g2 = 1 / (np.sqrt(2 * np.pi) * sigma2) * np.exp(-x ** 2 / (2 * sigma2 ** 2))
# g3 = g1-g2
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# plt.plot(x,g1, 'navy', label='Sigma = 1')
# plt.plot(x,g2, 'grey', label = 'Sigma = 0.5')
# plt.plot(x,g3, 'orange', label = 'DoG')
#
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.legend()
#
# plt.savefig('S:/06_Studienarbeit/06_Abschlussvortrag\Abbildungen/Difference_of_Gaussian.png')
#
# plt.show()
