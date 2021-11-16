import json
import os
import pprint
import unittest

import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from test_images import TestVideoFrame, plot_img
from test_images import test_data_path, img_length

from tools.graphs import get_positions_list, get_ext_adjacency_matrix
from tools.im2graph import sort_list_of_nodes
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


class TestSaveGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        list_of_nodes = [[7, 2.5], [2, 2], [4, 0], [0, 3], [0, 0]]

        cls.sorted_nodes = sort_list_of_nodes(list_of_nodes)
        cls.graph = cls.create_graph()

        cls.filepath = os.path.join(test_data_path, 'graph.json')

        assert not os.path.isfile(cls.filepath)

    @classmethod
    def create_graph(cls):
        graph = nx.Graph()

        list_of_edges = [[[0, 0], [2, 2]],
                         [[4, 0], [2, 2]],
                         [[0, 3], [2, 2]],
                         [[7, 2.5], [2, 2]]]
        edge_attrs = [1, 1, 1, 3]

        # define nodes with attribute position
        for i, xy in enumerate(cls.sorted_nodes):
            graph.add_node(i, pos=tuple(xy))

        # define edges with attributes: length
        for p, edge in enumerate(list_of_edges):
            start_xy, end_xy = edge

            startidx = cls.sorted_nodes.index(start_xy)
            endidx = cls.sorted_nodes.index(end_xy)

            # graph.add_edge(startidx, endidx, label=p)
            graph.add_edge(startidx, endidx, edge_id=p, length=edge_attrs[p])

        return graph

    def save_graph(self):
        # data_dict = nx.readwrite.json_graph.cytoscape_data(self.graph)
        # data_dict = nx.readwrite.json_graph.adjacency_data(self.graph)
        data_dict = nx.node_link_data(self.graph)
        pprint.pprint(data_dict)

        # {'directed': False,
        #  'graph': {},
        #  'links': [{'edge_id': 0, 'length': 1, 'source': 0, 'target': 2},
        #            {'edge_id': 2, 'length': 1, 'source': 1, 'target': 2},
        #            {'edge_id': 1, 'length': 1, 'source': 2, 'target': 3},
        #            {'edge_id': 3, 'length': 3, 'source': 2, 'target': 4}],
        #  'multigraph': False,
        #  'nodes': [{'id': 0, 'pos': (0, 0)},
        #            {'id': 1, 'pos': (0, 3)},
        #            {'id': 2, 'pos': (2, 2)},
        #            {'id': 3, 'pos': (4, 0)},
        #            {'id': 4, 'pos': (7, 2.5)}]}

        with open(self.filepath, 'w') as f:
            json.dump(data_dict, f)
        assert os.path.isfile(self.filepath)

    def test_save_and_read_graph_from_file(self):
        self.save_graph()
        with open(self.filepath, 'r') as f:
            data_dict = json.load(f)

        read_graph = nx.node_link_graph(data_dict)

        read_positions = get_positions_list(read_graph)
        read_adj_matr = nx.to_numpy_array(read_graph)
        read_length_matr = nx.to_numpy_array(read_graph, weight='length')

        true_adj_matr = [[0., 0., 1., 0., 0.],
                         [0., 0., 1., 0., 0.],
                         [1., 1., 0., 1., 1.],
                         [0., 0., 1., 0., 0.],
                         [0., 0., 1., 0., 0.]]
        true_length_matrix = [[0., 0., 1., 0., 0.],
                              [0., 0., 1., 0., 0.],
                              [1., 1., 0., 1., 3.],
                              [0., 0., 1., 0., 0.],
                              [0., 0., 3., 0., 0.]]

        self.assertTrue(self.sorted_nodes == read_positions)

        np.testing.assert_equal(true_adj_matr, read_adj_matr)
        np.testing.assert_equal(read_length_matr, true_length_matrix)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.isfile(cls.filepath):
            os.remove(cls.filepath)
