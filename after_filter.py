import os.path

import cv2
# import networkx as nx
import numpy as np
import sys

from functions.images import threshold_imgs
from functions.im2graph import preprocess, node_extraction, edge_extraction, helpernodes_BasicGraph_for_polyfit, \
    helpernodes_BasicGraph_for_structure
from functions.im2graph import polyfit_visualize, polyfit_training
from functions.im2graph import graph_extraction, graph_poly
from functions.im2graph import plot_graph_on_img_poly

from functions.images import apply_img_mask

from config import Config
from video_data import video_filepath, frequency, trim_times

import warnings

warnings.simplefilter('ignore', np.RankWarning)


def after_filter(conf):
    # apply mask
    apply_img_mask(conf)

    # thresholding
    threshold_imgs(conf)

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

        # skip already processed frames
        if os.path.isfile(filepath_overlay):
            continue

        thresholded = cv2.imread(filepath_thr, 0)

        # skeletonise
        edgelength = 10
        preprocessed_image = preprocess(thresholded, original, edgelength,
                                        conf.pr_plot, conf.pr_save, filepath_pr)

        # landmarks
        node_thick = 6
        bcnodes, bcnodescoor, endpoints, endpointscoor, allnodes, allnodescoor, marked_img = node_extraction(
            preprocessed_image, node_thick)

        edge_start_end, esecoor, edge_course, coordinates_global = edge_extraction(preprocessed_image, endpoints,
                                                                                   bcnodes)

        helperedges, ese_helperedges, helpernodescoor = helpernodes_BasicGraph_for_polyfit(coordinates_global, esecoor,
                                                                                           allnodescoor)

        helperedges_structure, ese_helperedges_structure, helpernodescoor_structure = helpernodes_BasicGraph_for_structure(
            coordinates_global, esecoor, allnodescoor, marked_img, conf.lm_plot, conf.lm_save, node_thick,
            filepath_lm)

        # polynomial
        visual_degree = 5
        point_density = 2
        cubic_thresh = 10  # deg3 > 10, otherwise only deg2 coefficients for training
        polyfit_coeff_visual, polyfit_coordinates, edge_coordinates, polyfit_points = polyfit_visualize(helperedges,
                                                                                                        ese_helperedges,
                                                                                                        visual_degree,
                                                                                                        point_density)

        training_parameters = polyfit_training(helperedges, ese_helperedges, cubic_thresh)

        deg3 = [item[0][0] for item in training_parameters]
        deg2 = [item[0][1] for item in training_parameters]
        edgelength = [item[1] for item in training_parameters]
        edgelength_bool = True
        deg3_bool = True
        deg2_bool = True
        label_selection = [edgelength_bool, deg3_bool, deg2_bool]
        graph = graph_extraction(helpernodescoor_structure, ese_helperedges_structure,
                                 deg3, deg2, edgelength, label_selection)

        # pos = nx.get_node_attributes(graph, 'pos')

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
        # node_thick = 6
        edge_thick = 2
        graph_poly(original, helpernodescoor, polyfit_coordinates, conf.poly_plot, conf.poly_save,
                   node_thick, edge_thick,
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
        # fig3 = plot_graph_on_img_straight(original, graph, node_size, node_color,
        #                                   edge_width, edge_color, fig3_plot, fig3_save,
        #                                   g_s_img_dir + g_s_img_name)

        # plot graph on image, polynom edges
        # node_thick = 6
        node_thick = 7
        edge_thick = 2
        plot_graph_on_img_poly(original, helpernodescoor, polyfit_coordinates,
                               conf.overlay_plot, conf.overlay_save,
                               node_thick, edge_thick, filepath_overlay)


if __name__ == '__main__':
    conf = Config(video_filepath, frequency, trim_times)
    after_filter(conf)
