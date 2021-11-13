import os

from test_images import TestVideoFrame, plot_img
from test_images import test_data_path, img_length

import networkx as nx
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tools.graphs import get_positions_list, get_ext_adjacency_matrix
from tools.images import extract_graph_and_helpers, generate_node_pos_img
from tools.plots import plot_graph_on_img_straight


class TestGraph(TestVideoFrame):
    @classmethod
    def setUpClass(cls) -> None:
        super(TestGraph, cls).setUpClass()

        cls.fp_adj_matrix = os.path.join(test_data_path, 'adj_matr.npy')
        assert not os.path.isfile(cls.fp_adj_matrix)

        cls.img_skel = cv2.imread(cls.img_skeletonised_fp, cv2.IMREAD_GRAYSCALE)
        plot_img(cls.img_skel)
        plt.show()

        cls.graph = cls.get_graph()

    @classmethod
    def get_graph(cls):
        graph, _, _, _ = extract_graph_and_helpers(cls.img_skel, '')
        return graph

    def test_generate_node_pos_img(self):
        self.assertIsNotNone(self.graph)

        pos = get_positions_list(self.graph)
        self.assertIsNotNone(pos)

        node_pos_img = generate_node_pos_img(self.graph, img_length)

        plot_img(node_pos_img)
        plt.show()

    def test_adjacency_matrix_skeletonised_match(self):
        pos = get_positions_list(self.graph)
        ext_adj_matr = get_ext_adjacency_matrix(self.graph,
                                                do_save=True,
                                                filepath=self.fp_adj_matrix)
        self.assertTrue(os.path.isfile(self.fp_adj_matrix))

        adj_matr = ext_adj_matr[0, :, 2:]
        plot_graph_on_img_straight(self.img_skel, pos, adj_matr)

    def test_adjacency_matrix(self):
        """ Test adjacency matrix with and without nodelist. """
        nodelist = sorted(self.graph.nodes())

        adj_matrix = nx.convert_matrix.to_numpy_matrix(self.graph)
        adj_matrix_nl = nx.convert_matrix.to_numpy_matrix(self.graph,
                                                          nodelist=nodelist)

        np.testing.assert_equal(adj_matrix, adj_matrix_nl)
        self.assertEqual(adj_matrix.shape[1], adj_matrix.shape[0])

    @classmethod
    def tearDownClass(cls) -> None:
        os.remove(cls.fp_adj_matrix)
