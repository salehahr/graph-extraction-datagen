import json
import os
import unittest
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tools.im2graph import (
    extract_graph,
    extract_nodes_and_edges,
    generate_graph,
    helper_polyfit,
    helper_structural_graph,
    polyfit_training,
)
from tools.images import create_mask
from tools.plots import plot_img
from tools.PolyGraph import PolyGraph

data_path = os.path.join(os.getcwd(), "../data/test")


def timer(func):
    def wrapper_timer(*args, **kwargs):
        t_start = time()
        fval = func(*args, **kwargs)
        t_end = time()

        t_elapsed = t_end - t_start
        print(f"{func.__name__} took {t_elapsed:.3f} s")

        return fval

    return wrapper_timer


class TestExtractEdges(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_path = os.path.join(data_path, "edges-0000_00400")

        cls.skeleton_fp = os.path.join(base_path, "skeleton.png")
        cls.simple_skeleton_fp = os.path.join(base_path, "simple-skeleton.png")
        cls.graph_fp = os.path.join(base_path, "graph.json")

        cls.img_skel = cv2.imread(cls.skeleton_fp, cv2.IMREAD_GRAYSCALE)
        cls.img_simple_skel = cv2.imread(cls.simple_skeleton_fp, cv2.IMREAD_GRAYSCALE)
        cls.graph = PolyGraph.load(cls.graph_fp)

        cls.num_edges = 18

    def test_extract_edges(self):
        """Ensures that the number of extracted edges matches the true number
        of edges in the graph."""
        nodes, edges = extract_nodes_and_edges(self.img_skel)

        helper_pf_edges, _ = helper_polyfit(nodes, edges)
        helper_sg_edges, _ = helper_structural_graph(nodes, edges)

        polyfit_params = polyfit_training(helper_pf_edges)
        graph = generate_graph(helper_sg_edges, nodes, polyfit_params, False, "")

        edges_extracted = edges["path"]
        edges_in_graph = graph.edges

        self.assertEqual(len(edges_extracted), len(edges_in_graph))


@unittest.skip("Not using RGB images yet.")
class TestMaskRGB(unittest.TestCase):
    """Tests three methods for loading a mask from file and masking
    a sample RGB photo (pre-cropped."""

    @classmethod
    def setUpClass(cls) -> None:
        # image
        fp_rgb = os.path.join(data_path, "synth-rgb.png")
        img_bgr = cv2.imread(fp_rgb, cv2.IMREAD_COLOR)
        cls.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cls.img_masked = None

        # filepaths
        cls.mask_indices_fp = os.path.join(os.getcwd(), "mask_invisible_indices.npy")
        cls.mask_indices_json = os.path.join(os.getcwd(), "mask_invisible_indices.json")
        cls.mask_pic_fp = os.path.join(os.getcwd(), "mask.png")

        # save files
        mask = create_mask(256)
        cls._generate_indices(mask)
        cls._generate_indices_json(mask)
        cls._generate_mask_pic(mask)

    def _plot(self, masked_rgb, title):
        mask = create_mask(256)

        plt.subplot(131)
        plot_img(mask, cmap="gray")
        plt.subplot(132)
        plot_img(self.img_rgb)
        plt.subplot(133)
        plot_img(masked_rgb)

        plt.suptitle(title)
        plt.show()

    @classmethod
    def _generate_indices(cls, mask):
        mask_invisible_indices = np.argwhere(mask == 0)
        np.save(cls.mask_indices_fp, mask_invisible_indices)

    @classmethod
    def _generate_indices_json(cls, mask):
        mask_invisible_indices = np.argwhere(mask == 0).tolist()
        jdump = json.dumps(mask_invisible_indices, cls=json.JSONEncoder)

        with open(cls.mask_indices_json, "w+") as f:
            json.dump(jdump, f)

    @classmethod
    def _generate_mask_pic(cls, mask):
        cv2.imwrite(cls.mask_pic_fp, mask)

    def setUp(self) -> None:
        self.img_masked = self.img_rgb.copy()

    @timer
    def test_multiply_mask(self):
        self.plot_title = "multiply"

        mask = cv2.imread(self.mask_pic_fp, cv2.IMREAD_GRAYSCALE)

        for i in range(3):
            self.img_masked[:, :, i] = np.multiply(mask, self.img_rgb[:, :, i])

    @timer
    def test_set_indices(self):
        self.plot_title = "set indices .npy"

        mask_invisible_indices = np.load(self.mask_indices_fp)

        for x, y in mask_invisible_indices:
            self.img_masked[x, y, :] = [0, 0, 0]

    @timer
    def test_set_indices_json(self):
        self.plot_title = "set indices .json"

        with open(self.mask_indices_json, "r+") as f:
            jdump = json.load(f)

        mask_invisible_indices = np.asarray(eval(jdump))

        for x, y in mask_invisible_indices:
            self.img_masked[x, y, :] = [0, 0, 0]

    def tearDown(self) -> None:
        self._plot(self.img_masked, self.plot_title)


class TestExtractGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_path = os.path.join(data_path, "extract_graph-GRK021-0000_26640")

        cls.skeleton_fp = os.path.join(base_path, "3-skeleton.png")

        cls.img_skel = cv2.imread(cls.skeleton_fp, cv2.IMREAD_GRAYSCALE)

    def test_extract_graph(self):
        graph, _, _, _ = extract_graph(self.img_skel, self.skeleton_fp, False)
        self.assertIsNotNone(graph)
