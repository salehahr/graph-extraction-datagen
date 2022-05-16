import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tests.test_images import RandomImage
from tools import EdgeExtractor, NodeContainer, NodeExtractor, legacy
from tools.Edge import flip_edge_coordinates
from tools.im2graph import edge_extraction
from tools.images import normalise, skeletonise_and_clean
from tools.plots import node_types_image, plot_bgr_img, plot_edges

data_path = os.path.join(os.getcwd(), "../data/test")


class TestLegacyFunctions(RandomImage):
    """
    Tests to compare the equivalence of the old and new functions.
    """

    @classmethod
    def setUpClass(cls) -> None:
        random = False

        if random:
            super(TestLegacyFunctions, cls).setUpClass()
        if not random:
            # base_path = os.path.join(data_path, "extract_nodes-GRK014-0002-08360")
            base_path = os.path.join(data_path, "extract_nodes-synth09-04560")
            cls.img_skeletonised_fp = os.path.join(base_path, "skeleton.png")
            cls.title = os.path.relpath(cls.img_skeletonised_fp, start=data_path)

        cls.img_skel = cv2.imread(cls.img_skeletonised_fp, cv2.IMREAD_GRAYSCALE)
        cls.plot_skeletonised()

        l_crossing_nodes, l_end_nodes, l_skel = legacy.node_extraction(cls.img_skel)
        cls.legacy_nodes = NodeContainer(l_crossing_nodes, l_end_nodes, [])
        cls.legacy_skel = normalise(l_skel)

        cls.legacy_se_yx, cls.legacy_paths_yx = legacy.edge_extraction(
            skeleton=cls.img_skel,
            endpoints=cls.legacy_nodes.end_nodes_yx + cls.legacy_nodes.border_nodes_yx,
            bcnodes=cls.legacy_nodes.crossing_nodes_yx,
        )

    @classmethod
    def plot_skeletonised(cls):
        plot_bgr_img(cls.img_skel, title=cls.title)
        plt.show()

    def test_node_extractor(self):
        ne = NodeExtractor(self.img_skel)
        self._compare_nodes(ne.img, ne.nodes)

    def _compare_nodes(self, skel: np.ndarray, nodes: NodeContainer):
        skel = normalise(skel)
        self._plot_comparison(skel, nodes)

        self.assertEqual(nodes.crossing_nodes_yx, self.legacy_nodes.crossing_nodes_yx)
        self.assertEqual(nodes.end_nodes_yx, self.legacy_nodes.end_nodes_yx)
        np.testing.assert_equal(skel, self.legacy_skel)

    def _plot_comparison(
        self,
        skel_new: np.ndarray,
        nodes_new: NodeContainer,
    ) -> None:
        plt.subplot(1, 3, 1)
        plot_bgr_img(self.legacy_skel)
        plt.title("Legacy")

        plt.subplot(1, 3, 2)
        plot_bgr_img(skel_new)
        plt.title("New")

        plt.subplot(1, 3, 3)
        plot_bgr_img(np.abs(self.legacy_skel - skel_new))
        plt.title("Difference")

        plt.suptitle(self.title)
        plt.show()

        node_img_legacy = node_types_image(self.legacy_nodes, skeleton=self.legacy_skel)
        node_img_new = node_types_image(nodes_new, skeleton=skel_new)
        plt.subplot(1, 2, 1)
        plot_bgr_img(node_img_legacy)
        plt.title("Legacy")

        plt.subplot(1, 2, 2)
        plot_bgr_img(node_img_new)
        plt.title("New")

        plt.suptitle(self.title)
        plt.show()

    def test_edge_extractor(self):
        # Currently there are discrepancies;
        # EdgeExtractor stops searching a path once it encounters an end/a crossing node,
        # also doesn't save edges with length 1.

        ee = EdgeExtractor(self.img_skel, self.legacy_nodes)
        plot_edges([self.legacy_paths_yx, ee.paths_yx])

        # cleaned (no edges with length 1)
        def print_warning(l_paths):
            num_warnings = 0
            for i, (path, l_path) in enumerate(zip(ee.paths_yx, l_paths)):
                if path != l_path and path not in l_paths:
                    num_warnings += 1
                    print(f"\nDiscrepancy at iter: {i}")
                    print(f"\tLegacy: {l_path}")
                    print(f"\tNew   : {path}")
            if num_warnings > 0:
                print(
                    f"+++ {num_warnings} WARNING{'S' if num_warnings > 1 else ''}! +++"
                )

        legacy_paths_cleaned = [p for p in self.legacy_paths_yx if len(p) > 2]
        diff_img_cleaned = plot_edges([legacy_paths_cleaned, ee.paths_yx], plot=False)
        print_warning(legacy_paths_cleaned)

        if ee.bad_nodes:
            print("Nodes to discard:")
            for n in ee.bad_nodes:
                print(n)

        bug_pixels = np.argwhere(diff_img_cleaned)
        if bug_pixels.any():
            print(f"Bug pixels: {bug_pixels}")

            # double check that the bad nodes are set to 0
            for n in ee.bad_nodes:
                diff_img_cleaned[n.row, n.col] = 0

        np.testing.assert_equal(diff_img_cleaned, np.zeros(diff_img_cleaned.shape))

    def test_edge_extraction(self):
        ee = edge_extraction(self.img_skel, self.legacy_nodes)
        se_yx = flip_edge_coordinates(ee["ese"])
        paths_yx = flip_edge_coordinates(ee["path"])

        plot_edges([self.legacy_paths_yx, paths_yx])

        self.assertEqual(se_yx, self.legacy_se_yx)
        self.assertEqual(paths_yx, self.legacy_paths_yx)


class TestCleanSkeleton(RandomImage):
    """
    Test to compare the equivalence of the old and new functions
    to get a skeletonised image from a thresholded image.
    """

    @classmethod
    def setUpClass(cls) -> None:
        base_path_skel = os.path.join(data_path, "synth01-0000_00040")
        thresh_fp = os.path.join(base_path_skel, "threshed.png")
        thresh_fp2 = os.path.join(base_path_skel, "threshed2.png")

        cls.img_thresh = cv2.imread(thresh_fp, cv2.IMREAD_GRAYSCALE)
        cls.img_thresh2 = cv2.imread(thresh_fp2, cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def _plot(img_thresh, img_cleaned, img_skel_orig):
        plt.subplot(131)
        plt.imshow(img_thresh, cmap="gray")
        plt.title("Thresholded")

        plt.subplot(132)
        plt.imshow(img_cleaned, cmap="gray")
        plt.title("Refactored")

        plt.subplot(133)
        plt.imshow(img_skel_orig, cmap="gray")
        plt.title("Legacy")

        plt.show()

    def test_clean(self):
        # Images from legacy function
        img_legacy = legacy.preprocess(self.img_thresh)
        img_legacy2 = legacy.preprocess(self.img_thresh2)

        # Images from refactored function
        img_refactored = skeletonise_and_clean(
            self.img_thresh, plot=False, save=False, directory=""
        )
        img_refactored2 = skeletonise_and_clean(
            self.img_thresh2, plot=False, save=False, directory=""
        )

        # Visual comparison, elementwise array comparison
        self._plot(self.img_thresh, img_refactored, img_legacy)
        # mismatched_indices = np.argwhere(np.not_equal(img_refactored, img_legacy))
        # d1 = os.path.join(data_path, "mod.png")
        # d2 = os.path.join(data_path, "orig.png")
        # cv2.imwrite(d1, img_refactored)
        # cv2.imwrite(d2, img_legacy)
        np.testing.assert_equal(img_refactored, img_legacy)

        self._plot(self.img_thresh2, img_refactored2, img_legacy2)
        # mismatched_indices2 = np.argwhere(np.not_equal(img_refactored2, img_legacy2))
        # d1 = os.path.join(data_path, "mod2.png")
        # d2 = os.path.join(data_path, "orig2.png")
        # cv2.imwrite(d1, img_refactored2)
        # cv2.imwrite(d2, img_legacy2)
        np.testing.assert_equal(img_refactored2, img_legacy2)
