import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from test_images import ImageWithBorderNodes
from plots import plot_bgr_img
from tools.plots import node_types_image
from tools.im2graph import extract_nodes_and_edges
from tools.NodeContainer import NodeContainer, get_border_coordinates

test_data_path = '/graphics/scratch/schuelej/sar/graph-training/data/test'

img_length = 256
base_path = f'/graphics/scratch/schuelej/sar/data/{img_length}'


class TestExtractNodes(ImageWithBorderNodes):
    """
    Checks functions of the NodeContainer class for an image which
    has nodes on the border.
    Note: helper nodes aren't couunted in this test.
    """
    @classmethod
    def setUpClass(cls) -> None:
        super(TestExtractNodes, cls).setUpClass()
        cls.nodes = cls.extract_nodes()

    @classmethod
    def extract_nodes(cls):
        nodes, _, _ = extract_nodes_and_edges(cls.img_skel)
        assert isinstance(nodes, NodeContainer)
        return nodes

    def test_get_border_coordinates(self):
        """
        Visual test: reconstruct the border from the coordinates.
        """
        border_coords = get_border_coordinates()
        self.assertIsNotNone(border_coords)

        border_img = np.zeros((img_length, img_length, 3)).astype(np.float32)
        bgr_red = (0, 0, 255)

        for xy in border_coords:
            cv2.circle(border_img, xy, 1, bgr_red, -1)

        plot_bgr_img(border_img)
        plt.show()

    def test_extract_nodes(self):
        self.assertIsNotNone(self.nodes.border_nodes_yx)
        self.assertIsNotNone(self.nodes.end_nodes_yx)
        self.assertIsNotNone(self.nodes.crossing_nodes_xy)
        self.assertEqual(self.nodes.num_all_nodes,
                         self.nodes.num_crossing_nodes
                         + self.nodes.num_end_nodes
                         + self.nodes.num_border_nodes)

    def test_classify_nodes(self):
        """
        Visual test for checking that the border nodes are classified correctly.
        """
        new_lm_img = node_types_image(self.nodes, image_length=img_length)

        plot_bgr_img(new_lm_img)
        plt.show()


class TestNodeContainerFromGraph(TestExtractNodes):
    @classmethod
    def extract_nodes(cls):
        print('TestNodeContainerFromGraph.extract_nodes class method')
        graph_fp = cls.img_skel_fp.replace('skeleton', 'graphs').replace('.png', '.json')

        return NodeContainer(graph_fp=graph_fp)
