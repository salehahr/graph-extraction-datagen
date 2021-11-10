import os
import unittest

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from config import Config
from functions.graphs import get_positions_list, get_adjacency_matrix
from functions.images import crop_resize_square, is_square, crop_radius, generate_node_pos_img
from functions.images import get_rgb, get_centre
from functions.images import extract_graph_and_helpers


base_path = '/graphics/scratch/schuelej/sar/graph-training/data/test'


def plot_img(img):
    rgb_img = get_rgb(img)
    plt.imshow(rgb_img)
    plt.xticks([])
    plt.yticks([])


class TestShortVideoImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        filename = os.path.join(base_path, 'short_video.mp4')
        cls.config = Config(filename, frequency=2, img_length=512, trim_times=None)

        first_img_fp = cls.config.raw_image_files[0]
        cls.img = cv2.imread(first_img_fp, cv2.IMREAD_COLOR)

        img_skel_fp = cls.config.skeletonised_image_files[0]
        cls.img_skel = cv2.imread(img_skel_fp, cv2.IMREAD_GRAYSCALE)

        cls.graph = cls.get_graph()

    @classmethod
    def get_graph(cls):
        cls.config.lm_save = False
        graph, _, _, _ = extract_graph_and_helpers(cls.config, cls.img_skel, '')
        return graph

    def test_trimmed(self):
        self.assertFalse(self.config.has_trimmed)
        self.assertFalse(self.config.is_trimmed)

    def test_find_centre(self):
        cx, cy = get_centre(self.img)
        cv2.circle(self.img, (cx, cy), int(crop_radius / 2), (10, 80, 10), -1)

        plt.figure()
        plot_img(self.img)
        plt.show()

    def test_crop(self):
        height, width, _ = self.img.shape

        self.assertEqual(height, 1080)
        self.assertEqual(width, 1920)

        square_img = crop_resize_square(self.img, self.config.img_length)
        cr_height, cr_width, _ = square_img.shape

        # plt.figure()
        # plot_img(square_img)
        # plt.show()

        self.assertTrue(is_square(square_img))
        self.assertEqual(self.config.img_length, cr_height)
        self.assertEqual(self.config.img_length, cr_width)

    def test_generate_node_pos_img(self):
        self.assertIsNotNone(self.graph)

        pos = get_positions_list(self.graph)
        self.assertIsNotNone(pos)

        node_pos_img = generate_node_pos_img(self.config, pos)

        temp_fp_skeleton = os.path.join(base_path, 'temp_skel.png')
        temp_fp_nodepos = os.path.join(base_path, 'temp_nodepos.png')
        cv2.imwrite(temp_fp_skeleton, self.img_skel)
        cv2.imwrite(temp_fp_nodepos, node_pos_img)

    def test_get_adjacency_matrix(self):
        fp_adj_matrix = os.path.join(base_path, 'adj_matr.npy')
        extended_adj_matrix = get_adjacency_matrix(self.graph,
                                                   do_save=True,
                                                   filepath=fp_adj_matrix)

        self.assertTrue(os.path.isfile(fp_adj_matrix))

        with open(fp_adj_matrix, 'rb') as f:
            a = np.load(f)
            self.assertEqual(a.shape[0], len(self.graph))
            self.assertEqual(a.shape[1], 2 + len(self.graph))

        os.remove(fp_adj_matrix)
        self.assertFalse(os.path.isfile(fp_adj_matrix))

    def test_adjacency_matrix(self):
        """ Test adjacency matrix with and without nodelist. """
        nodelist = sorted(self.graph.nodes())

        adj_matrix = nx.convert_matrix.to_numpy_matrix(self.graph)
        adj_matrix_nl = nx.convert_matrix.to_numpy_matrix(self.graph,
                                                          nodelist=nodelist)

        np.testing.assert_equal(adj_matrix, adj_matrix_nl)
        self.assertEqual(adj_matrix.shape[1], adj_matrix.shape[0])
