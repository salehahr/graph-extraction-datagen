import os
import unittest

import cv2

from config import Config
from functions.images import crop_resize_square, is_square, crop_radius
from functions.images import get_rgb, get_centre

import matplotlib.pyplot as plt

base_path = '/graphics/scratch/schuelej/sar/graph-training/data'


def plot_img(img):
    rgb_img = get_rgb(img)
    plt.imshow(rgb_img)
    plt.xticks([])
    plt.yticks([])


class TestResize(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        filename = os.path.join(base_path, 'test/short_video.mp4')
        cls.config = Config(filename, frequency=2, img_length=512, trim_times=None)

        first_img_fp = cls.config.raw_image_files[0]
        cls.img = cv2.imread(first_img_fp, cv2.IMREAD_COLOR)

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
