import os

from test_images import TestVideoFrame, plot_img
from test_images import test_data_path, img_length

import networkx as nx
import numpy as np
import cv2
import matplotlib.pyplot as plt

from functions.graphs import get_positions_list, get_adjacency_matrix
from functions.images import extract_graph_and_helpers, generate_node_pos_img


class TestGraph(TestVideoFrame):
    @classmethod
    def setUpClass(cls) -> None:
        super(TestGraph, cls).setUpClass()

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
        fp_adj_matrix = os.path.join(test_data_path, 'adj_matr.npy')
        self.assertFalse(os.path.isfile(fp_adj_matrix))

        get_adjacency_matrix(self.graph, do_save=True, filepath=fp_adj_matrix)
        self.assertTrue(os.path.isfile(fp_adj_matrix))

        with open(fp_adj_matrix, 'rb') as f:
            a = np.load(f)
            print(a.shape)
            self.assertEqual(a.shape[1], len(self.graph))
            self.assertEqual(a.shape[2], 2 + len(self.graph))

        os.remove(fp_adj_matrix)

    def test_adjacency_matrix(self):
        """ Test adjacency matrix with and without nodelist. """
        nodelist = sorted(self.graph.nodes())

        adj_matrix = nx.convert_matrix.to_numpy_matrix(self.graph)
        adj_matrix_nl = nx.convert_matrix.to_numpy_matrix(self.graph,
                                                          nodelist=nodelist)

        np.testing.assert_equal(adj_matrix, adj_matrix_nl)
        self.assertEqual(adj_matrix.shape[1], adj_matrix.shape[0])
