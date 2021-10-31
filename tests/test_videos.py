import glob
import os
import unittest

from functions_files import make_folders, delete_files
from functions_videos import video2img, trim_video
from functions_images import crop_imgs

from config import Config


class TestVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = None
        cls.raw_img_folder = None

    @property
    def list_img_files(self):
        return glob.glob(os.path.join(self.raw_img_folder, '*.png'))

    def before_filter(self):
        make_folders(self.config)
        video2img(self.config, frequency=2)
        crop_imgs(self.config)

    def test_video2img(self):
        if self.config:
            delete_files(self.list_img_files)
            self.assertEqual(len(self.list_img_files), 0)

            self.before_filter()

            self.assertGreaterEqual(len(self.list_img_files), 1)


class TestVideoLocal(TestVideo):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config('short_video.mp4')
        cls.raw_img_folder = cls.config.raw_img_folder

    def test_is_not_trimmed(self):
        self.assertFalse(self.config.is_trimmed)


class TestTrimmedVideo(TestVideo):
    @classmethod
    def setUpClass(cls) -> None:
        cls.video_filename = "trimmed_0000_02000__0000_03000.mp4"
        cls.config = Config(cls.video_filename)
        cls.raw_img_folder = cls.config.raw_img_folder

    def test_is_trimmed(self):
        self.assertTrue(self.config.is_trimmed)


# class TestVideoNetwork(TestVideo):
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.config = Config('M:/ma/graph-training/data/short_video.mp4')
#         cls.raw_img_folder = cls.config.raw_img_folder


class TestTrimVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.orig_filename = "../data/GRK021_test.mp4"
        cls.target_filename = "trimmed_0000_02000__0000_03000.mp4"

    def test_trim_video(self):
        self.assertTrue(os.path.isfile(self.orig_filename))
        os.remove(self.target_filename)
        self.assertFalse(os.path.isfile(self.target_filename))

        start_time_in_s = 2
        end_time_in_s = 3
        trim_video(self.orig_filename, start_time_in_s, end_time_in_s, self.target_filename)

        self.assertTrue(os.path.isfile(self.target_filename))
