import os
import cv2
import numpy as np

from tools.graphs import extract_nodes_edges, get_positions_vector, get_positions_list, get_ext_adjacency_matrix

from tools.im2graph import preprocess, polyfit_training
from tools.im2graph import helpernodes_BasicGraph_for_polyfit
from tools.im2graph import polyfit_visualize, graph_extraction, graph_poly, \
    plot_graph_on_img_poly

blur_kernel = (5, 5)
crop_radius = 575


def get_rgb(img):
    """ Gets RGB image for matplotlib plots. """
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def is_square(img: np.ndarray):
    h, w, _ = img.shape
    return h == w


def get_centre(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_thresholded():
        # create background image
        bg = cv2.dilate(img_grey, np.ones(blur_kernel, dtype=np.uint8))
        bg = cv2.GaussianBlur(bg, blur_kernel, 1)

        # subtract out background from source
        src_no_bg = 255 - cv2.absdiff(img_grey, bg)

        # threshold
        _, thresh = cv2.threshold(src_no_bg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return thresh

    # calculate moments of binary image
    moments = cv2.moments(get_thresholded())

    # calculate x,y coordinate of center
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def crop_imgs(conf):
    for fp in conf.raw_image_files:
        new_fp = fp.replace('raw', 'cropped')

        if os.path.isfile(new_fp):
            continue

        img = cv2.imread(fp, cv2.IMREAD_COLOR)

        if is_square(img):
            continue

        img_cropped = crop_resize_square(img, conf.img_length)

        # cv2.imshow('title', img_cropped)
        # cv2.waitKey()

        cv2.imwrite(new_fp, img_cropped)


def crop_resize_square(img: np.ndarray, length: int):
    return resize_square(centre_crop(img), length)


def resize_square(img: np.ndarray, length: int):
    return cv2.resize(img, (length, length))


def centre_crop(img: np.ndarray):
    """
    Extracts a square from the original image,
    from around the original image's centroid.
    """
    min_y, min_x = 0, 0
    max_y, max_x, _ = img.shape

    cx, cy = get_centre(img)

    # get crop dimensions
    crop_left = cx - crop_radius
    crop_right = cx + crop_radius
    crop_top = cy - crop_radius
    crop_bottom = cy + crop_radius

    # calculate padding
    top_pad = 0 if crop_top >= 0 else -crop_top
    bottom_pad = 0 if crop_bottom <= max_y else crop_bottom - max_y
    left_pad = 0 if crop_left >= 0 else -crop_left
    right_pad = 0 if crop_right <= max_x else crop_right - max_x

    # set crop dimensions to max edges if necessary
    crop_left = crop_left if crop_left >= 0 else min_x
    crop_right = crop_right if crop_right <= max_x else max_x
    crop_top = crop_top if crop_top >= 0 else min_y
    crop_bottom = crop_bottom if crop_bottom <= max_y else max_y

    img = img[crop_top:crop_bottom, crop_left:crop_right]

    return cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])


def apply_img_mask(conf):
    mask_path = f'../data/mask{conf.img_length:d}.png' if 'tests' in os.getcwd() \
        else f'data/mask{conf.img_length:d}.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255

    for fp in conf.filtered_image_files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        masked = np.multiply(mask, img)

        # cv2.imshow('title', masked_img)
        # cv2.waitKey()

        new_fp = fp.replace('filtered', 'masked')
        cv2.imwrite(new_fp, masked)


def threshold_imgs(conf):
    for fp in conf.masked_image_files:
        new_fp = fp.replace('masked', 'threshed')
        img = cv2.imread(fp, 0)
        thresholding(img, conf.thr_save, new_fp)


def thresholding(filtered_img: np.ndarray, do_save: bool, filepath: str = '') \
        -> np.ndarray:
    blurred_img = cv2.GaussianBlur(filtered_img, blur_kernel, 0)
    _, thresholded_img = cv2.threshold(blurred_img, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if do_save:
        cv2.imwrite(filepath, thresholded_img)

    return thresholded_img


def skeletonise_imgs(conf):
    for fp in conf.threshed_image_files:
        skel_fp = fp.replace('threshed', 'skeleton')

        img_threshed = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)

        preprocess(img_threshed,
                   conf.pr_plot, conf.pr_save, skel_fp)


def extract_graphs(conf, skip_existing):
    """ Starting with the thresholded images, performs the operations
    skeletonise, node extraction, edge extraction"""

    for fp in conf.skeletonised_image_files:
        cropped_fp = fp.replace('skeleton', 'cropped')
        overlay_fp = fp.replace('skeleton', 'overlay')
        poly_fp = fp.replace('skeleton', 'poly_graph')

        node_pos_img_fp = fp.replace('skeleton', 'node_positions')
        node_pos_vec_fp = os.path.splitext(fp.replace('skeleton', 'node_positions'))[0] + '.npy'
        adj_matr_fp = os.path.splitext(fp.replace('skeleton', 'adj_matr'))[0] + '.npy'

        img_cropped = cv2.imread(cropped_fp, cv2.IMREAD_COLOR)
        img_preproc = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)

        # exit if no raw image found
        if img_preproc is None:
            print(f'No skeletonised image found.')
            raise Exception

        # skip already processed frames
        if os.path.isfile(overlay_fp) and skip_existing:
            continue

        # landmarks
        graph, ese_h_edges, h_edges, h_edges_cds = extract_graph_and_helpers(img_preproc,
                                                                             fp,
                                                                             conf.lm_plot,
                                                                             conf.lm_save,
                                                                             conf.graph_save)

        if conf.node_pos_save or conf.adj_matr_save:
            get_positions_vector(graph,
                                 do_save=conf.node_pos_save, filepath=node_pos_vec_fp)
            generate_node_pos_img(graph, conf.img_length,
                                  do_save=conf.node_pos_save, filepath=node_pos_img_fp)
            get_ext_adjacency_matrix(graph,
                                     do_save=conf.adj_matr_save, filepath=adj_matr_fp)

        # plot polynomials
        visualise_poly = conf.poly_plot or conf.poly_save
        visualise_overlay = conf.overlay_plot or conf.overlay_save

        if visualise_poly or visualise_overlay:
            edge_width = 2
            _, polyfit_coordinates, _, _ = polyfit_visualize(h_edges, ese_h_edges)

            if visualise_poly:
                node_size = 10
                graph_poly(img_cropped, h_edges_cds, polyfit_coordinates, conf.poly_plot, conf.poly_save,
                           node_size, edge_width, poly_fp)

            if visualise_overlay:
                node_size = 7
                plot_graph_on_img_poly(img_cropped, h_edges_cds, polyfit_coordinates,
                                       conf.overlay_plot, conf.overlay_save,
                                       node_size, edge_width, overlay_fp)


def extract_graph_and_helpers(img_preproc, skel_fp, lm_plot=False, lm_save=False, graph_save=False):
    landmarks_fp = skel_fp.replace('skeleton', 'landmarks')
    graph_fp = os.path.splitext(skel_fp.replace('skeleton', 'graphs'))[0] + '.json'

    node_size = 6
    allnodes_xy, edge_course_xy, ese_xy, img_lm = extract_nodes_edges(img_preproc,
                                                                      node_size)
    helperedges, ese_helperedges, helpernodescoor = helpernodes_BasicGraph_for_polyfit(edge_course_xy,
                                                                                       ese_xy,
                                                                                       allnodes_xy)

    training_parameters = polyfit_training(helperedges, ese_helperedges)
    graph = graph_extraction(edge_course_xy,
                             ese_xy,
                             allnodes_xy,
                             img_lm,
                             lm_plot,
                             lm_save,
                             node_size,
                             landmarks_fp,
                             training_parameters,
                             graph_save,
                             graph_fp)

    return graph, ese_helperedges, helperedges, helpernodescoor


def generate_node_pos_img(graph, dim, do_save: bool = True, filepath: str = ''):
    img = np.zeros((dim, dim)).astype(np.uint8)

    node_positions = get_positions_list(graph)
    for coords in node_positions:
        x, y = coords
        row, col = y, x
        img[row][col] = 255

    if do_save and filepath:
        cv2.imwrite(filepath, img)

    return img
