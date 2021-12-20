import os
import unittest

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tools.files import get_random_raw_image, get_random_video_path
from tools.images import create_mask, crop_resize_square, get_centre
from tools.plots import plot_bgr_img, plot_border_overlay

data_path = os.path.join(os.getcwd(), "../data/test")


def is_square(img: np.ndarray):
    h, w, _ = img.shape
    return h == w


img_length = 256
base_path = f"/graphics/scratch/schuelej/sar/data/{img_length}"


class RandomImage(unittest.TestCase):
    """Class to get a random image from the data folder for testing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_path = base_path
        cls.video_path = get_random_video_path(base_path)

        img_name = get_random_raw_image(cls.video_path)
        print(f"Video path: {cls.video_path}")
        print(f"Image: {img_name}")

        cls.is_synthetic = "synthetic" in cls.video_path
        cls.title = os.path.join(
            os.path.relpath(cls.video_path, start=base_path),
            os.path.splitext(img_name)[0],
        )

        cls.img_raw_fp = os.path.join(cls.video_path, f"raw/{img_name}")
        cls.img_skeletonised_fp = cls.img_raw_fp.replace("raw", "skeleton")

        cls.img_length = img_length

        assert os.path.isfile(cls.img_raw_fp)
        assert os.path.isfile(cls.img_skeletonised_fp)

        cls.plot_cropped()

    @classmethod
    def plot_cropped(cls):
        cropped_fp = cls.img_raw_fp.replace("raw", "cropped")
        img_cropped = cv2.imread(cropped_fp, cv2.IMREAD_COLOR)
        plot_bgr_img(img_cropped)
        plt.title(cls.title)
        plt.show()


class ImageWithBorderNodes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = os.path.join(data_path, "border_nodes_GRK011_0002_20039")

        cls.img_skel_fp = os.path.join(path, "skeleton.png")
        cls.img_skel = cv2.imread(cls.img_skel_fp, cv2.IMREAD_GRAYSCALE)

        img_landmarks_fp = cls.img_skel_fp.replace("skeleton", "landmarks")
        plot_border_overlay(img_landmarks_fp)


class NaNImage(unittest.TestCase):
    """
    Image that causes a NaN in polyfit_training.
    """

    @classmethod
    def setUpClass(cls) -> None:
        nan_img_name = "0000_14120.png"
        nan_video_path = os.path.join(base_path, "GRK008")

        cls.img_cropped_fp = os.path.join(nan_video_path, f"cropped/{nan_img_name}")
        cls.img_skel_fp = os.path.join(nan_video_path, f"skeleton/{nan_img_name}")
        cls.img_skel = cv2.imread(cls.img_skel_fp, cv2.IMREAD_GRAYSCALE)

        img_landmarks_fp = os.path.join(nan_video_path, f"landmarks/{nan_img_name}")
        plot_border_overlay(img_landmarks_fp)


class TestImage(RandomImage):
    @classmethod
    def setUpClass(cls) -> None:
        super(TestImage, cls).setUpClass()

        img_raw = cv2.imread(cls.img_raw_fp, cv2.IMREAD_COLOR)

        cls.crop_radius = int(img_raw.shape[0] / 2) if cls.is_synthetic else 575
        cls.img_raw = img_raw

    def test_find_centre(self):
        if self.is_synthetic:
            return

        cx, cy = get_centre(self.img_raw)
        cv2.circle(self.img_raw, (cx, cy), int(self.crop_radius / 2), (10, 80, 10), -1)

        plt.figure()
        plot_bgr_img(self.img_raw)
        plt.show()

    def test_crop(self):
        square_img = crop_resize_square(
            self.img_raw, self.img_length, self.is_synthetic
        )
        cr_height, cr_width, _ = square_img.shape

        self.assertTrue(is_square(square_img))
        self.assertEqual(self.img_length, cr_height)
        self.assertEqual(self.img_length, cr_width)

    @unittest.skip("Now irrelevant.")
    def test_apply_new_mask(self):
        """
        Old mask was a graphic created in Inkscape and imported as a numpy file.
        New mask is created using the cv2.circle function
        """
        old_mask_fp = (
            f"/graphics/scratch/schuelej/sar/graph-training/data/mask{img_length:d}.png"
        )
        self.assertTrue(os.path.isfile(old_mask_fp))

        old_mask = cv2.imread(old_mask_fp, cv2.IMREAD_GRAYSCALE) / 255
        new_mask = create_mask(img_length)

        filtered_fp = self.img_raw_fp.replace("raw", "filtered")
        img_filtered = cv2.imread(filtered_fp, cv2.IMREAD_GRAYSCALE)
        plot_bgr_img(img_filtered, title="filtered")

        img_masked_old = np.multiply(old_mask, img_filtered).astype(np.uint8)
        img_masked_new = np.multiply(new_mask, img_filtered).astype(np.uint8)
        plot_bgr_img(img_masked_old, title="masked old")
        plot_bgr_img(img_masked_new, title="masked new")
        plt.show()

        # goal: less than 1% mismatch in pixels
        mismatches = np.where(img_masked_old != img_masked_new)
        num_mismatches = len(mismatches[0])
        total_elements = img_masked_old.size
        fraction_mismatches = num_mismatches / total_elements

        print(
            f"{num_mismatches} mismatches out of {total_elements} elements, "
            f"{fraction_mismatches * 100:.2f} %"
        )

        self.assertLessEqual(fraction_mismatches, 1)
