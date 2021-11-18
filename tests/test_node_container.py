import os
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np

from test_images import plot_img
from tools.images import overlay_border, node_types_image
from tools.im2graph import extract_graph_and_helpers
from tools.NodeContainer import NodeContainer, get_border_coordinates

test_data_path = '/graphics/scratch/schuelej/sar/graph-training/data/test'

img_length = 256
base_path = f'/graphics/scratch/schuelej/sar/data/{img_length}'
video_path = os.path.join(base_path, 'GRK011/0002_17000__0002_21000')


def plot_overlay(img_lm_fp):
    """
    Plot a circular border on the landmarks image.
    """
    img_lm_border = cv2.imread(img_lm_fp, cv2.IMREAD_COLOR)
    overlay_border(img_lm_border)

    plot_img(img_lm_border, title='landmarks')
    plt.show()


class TestBorderCase(unittest.TestCase):
    """
    Checks functions of the NodeContainer class for an image which
    has nodes on the border.
    """
    @classmethod
    def setUpClass(cls) -> None:
        img_name = '0002_20039.png'
        cls.img_cropped_fp = os.path.join(video_path, f'cropped/{img_name}')
        cls.img_skel_fp = os.path.join(video_path, f'skeleton/{img_name}')
        cls.img_skel = cv2.imread(cls.img_skel_fp, cv2.IMREAD_GRAYSCALE)

        img_landmarks_fp = os.path.join(video_path, f'landmarks/{img_name}')
        plot_overlay(img_landmarks_fp)

        cls.nodes = cls.extract_nodes()

    @classmethod
    def extract_nodes(cls):
        _, nodes, _, _, _ = extract_graph_and_helpers(cls.img_skel, '')
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

        plot_img(border_img)
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
        new_lm_img = node_types_image(img_length, self.nodes)

        plot_img(new_lm_img)
        plt.show()


class TestNodeContainerFromGraph(TestBorderCase):
    @classmethod
    def extract_nodes(cls):
        print('TestNodeContainerFromGraph.extract_nodes class method')
        graph_fp = cls.img_skel_fp.replace('skeleton', 'graphs').replace('.png', '.json')

        return NodeContainer(graph_fp=graph_fp)
