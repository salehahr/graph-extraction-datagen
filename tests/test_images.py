import os
import unittest

import cv2
import matplotlib.pyplot as plt
from functions.images import crop_resize_square, is_square, crop_radius
from functions.images import get_rgb, get_centre


img_length = 256
data_path = f'/graphics/scratch/schuelej/sar/data/{img_length}/GRK008/raw'
test_data_path = '/graphics/scratch/schuelej/sar/graph-training/data/test'


def plot_img(img):
    rgb_img = get_rgb(img)
    plt.imshow(rgb_img)
    plt.xticks([])
    plt.yticks([])


class TestVideoFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.img_raw_fp = os.path.join(data_path, '0000_00000.png')
        cls.img_skeletonised_fp = cls.img_raw_fp.replace('raw', 'skeleton')

        cls.img_length = img_length

        assert os.path.isfile(cls.img_raw_fp)
        assert os.path.isfile(cls.img_skeletonised_fp)


class TestImage(TestVideoFrame):
    @classmethod
    def setUpClass(cls) -> None:
        super(TestImage, cls).setUpClass()

        cls.img_raw = cv2.imread(cls.img_raw_fp, cv2.IMREAD_COLOR)
        cls.img_skel = cv2.imread(cls.img_skeletonised_fp, cv2.IMREAD_GRAYSCALE)

    def test_find_centre(self):
        cx, cy = get_centre(self.img_raw)
        cv2.circle(self.img_raw, (cx, cy), int(crop_radius / 2), (10, 80, 10), -1)

        plt.figure()
        plot_img(self.img_raw)
        plt.show()

    def test_crop(self):
        height, width, _ = self.img_raw.shape

        self.assertEqual(height, 1080)
        self.assertEqual(width, 1920)

        square_img = crop_resize_square(self.img_raw, self.img_length)
        cr_height, cr_width, _ = square_img.shape

        # plt.figure()
        # plot_img(square_img)
        # plt.show()

        self.assertTrue(is_square(square_img))
        self.assertEqual(self.img_length, cr_height)
        self.assertEqual(self.img_length, cr_width)

