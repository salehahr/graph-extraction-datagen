import os
import unittest

import cv2
import matplotlib.pyplot as plt

from tools.files import get_random_video_path, get_random_raw_image
from tools.images import crop_resize_square, is_square, crop_radius
from tools.images import get_rgb, get_centre


def plot_img(img):
    rgb_img = get_rgb(img)
    plt.imshow(rgb_img)
    plt.xticks([])
    plt.yticks([])


img_length = 256
test_data_path = '/graphics/scratch/schuelej/sar/graph-training/data/test'
base_path = f'/graphics/scratch/schuelej/sar/data/{img_length}'
video_path = get_random_video_path(base_path)


class TestVideoFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base_path = base_path
        cls.video_path = video_path

        img_name = get_random_raw_image(cls.video_path)
        print(f'Video path: {cls.video_path}')
        print(f'Image: {img_name}')

        cls.img_raw_fp = os.path.join(cls.video_path, f'raw/{img_name}')
        cls.img_skeletonised_fp = cls.img_raw_fp.replace('raw', 'skeleton')

        cls.img_length = img_length

        assert os.path.isfile(cls.img_raw_fp)
        assert os.path.isfile(cls.img_skeletonised_fp)

        cls.plot_cropped()

    @classmethod
    def plot_cropped(cls):
        cropped_fp = cls.img_raw_fp.replace('raw', 'cropped')
        img_cropped = cv2.imread(cropped_fp, cv2.IMREAD_COLOR)
        plot_img(img_cropped)
        plt.title(os.path.relpath(cls.img_raw_fp, start=base_path))
        plt.show()


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

