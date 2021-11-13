import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from test_images import TestVideoFrame, plot_img
from test_images import test_data_path, img_length

from tools.graphs import get_positions_list, get_ext_adjacency_matrix
from tools.images import extract_graph_and_helpers, generate_node_pos_img
from tools.plots import plot_graph_on_img_straight


class TestGraph(TestVideoFrame):
    """
    Sanity checks:
        * tests whether the node positions are sorted
        * tests whether the adjacency matrix matches the skeletonised image
    """
    @classmethod
    def setUpClass(cls) -> None:
        super(TestGraph, cls).setUpClass()

        cls.img_skel = cv2.imread(cls.img_skeletonised_fp, cv2.IMREAD_GRAYSCALE)
        cls.plot_skeletonised()

        cls.pos_list = None

    @classmethod
    def plot_skeletonised(cls):
        plot_img(cls.img_skel)
        plt.title(os.path.relpath(cls.img_skeletonised_fp, start=cls.base_path))
        plt.show()

    def plot_adj_matr(self, adj_matr):
        plot_graph_on_img_straight(self.img_skel, self.pos_list, adj_matr)

    def test_is_pos_list_sorted(self):
        rows, cols = zip(*self.pos_list)
        print(self.pos_list)

        def is_sorted_ascending(arr):
            return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))

        is_sorted_row = is_sorted_ascending(rows)
        is_sorted_col = is_sorted_ascending(cols)

        self.assertTrue(is_sorted_row or is_sorted_col)

    def test_adjacency_matrix_skeletonised_match(self):
        pass


class TestReadGraph(TestGraph):
    """
    Implements TestGraph checks for pre-generated (already saved)
    adjacency matrix and node positions of a random image.
    """
    @classmethod
    def setUpClass(cls) -> None:
        super(TestReadGraph, cls).setUpClass()

        fp = cls.img_skeletonised_fp
        node_pos_vec_fp = os.path.splitext(fp.replace('skeleton', 'node_positions'))[0] + '.npy'
        adj_matr_fp = os.path.splitext(fp.replace('skeleton', 'adj_matr'))[0] + '.npy'

        ext_adj_matr = np.load(adj_matr_fp)
        cls.adj_matr = ext_adj_matr[0, :, 2:].astype(np.uint8)

        node_pos = np.load(node_pos_vec_fp).astype(np.uint8)
        cls.pos_list = np.ndarray.tolist(node_pos)

    def test_adjacency_matrix_skeletonised_match(self):
        self.plot_adj_matr(self.adj_matr)


class TestGenerateGraph(TestGraph):
    """
    Implements TestGraph checks for
    adjacency matrix and node positions which are generated on the fly.
    """
    @classmethod
    def setUpClass(cls) -> None:
        super(TestGenerateGraph, cls).setUpClass()

        cls.fp_adj_matrix = os.path.join(test_data_path, 'adj_matr.npy')
        if os.path.isfile(cls.fp_adj_matrix):
            os.remove(cls.fp_adj_matrix)

        cls.graph, _, _, _ = extract_graph_and_helpers(cls.img_skel, '')
        cls.pos_list = get_positions_list(cls.graph)

    def test_generate_node_pos_img(self):
        node_pos_img = generate_node_pos_img(self.graph, img_length)

        plot_img(node_pos_img)
        plt.show()

    def test_adjacency_matrix_skeletonised_match(self):
        ext_adj_matr = get_ext_adjacency_matrix(self.graph,
                                                do_save=True,
                                                filepath=self.fp_adj_matrix)
        self.assertTrue(os.path.isfile(self.fp_adj_matrix))

        adj_matr = ext_adj_matr[0, :, 2:]
        self.plot_adj_matr(adj_matr)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            os.remove(cls.fp_adj_matrix)
        except FileNotFoundError:
            pass
