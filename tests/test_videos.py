import glob
import os
import unittest

from functions_files import make_folders, delete_files
from functions_videos import video2img, trim_video
from config import Config


class TestVideoLocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = Config('short_video.mp4')
        cls.raw_img_folder = cls.config.raw_img_folder
        cls.video_filename = cls.config.filepath
        cls.list_of_folders = cls.config.list_of_folders

    @property
    def list_img_files(self):
        return glob.glob(os.path.join(self.raw_img_folder, '*.png'))

    def test_is_not_trimmed(self):
        self.assertFalse(self.config.is_trimmed)

    def test_video2img(self):
        make_folders(self.list_of_folders)
        delete_files(self.list_img_files)
        self.assertEqual(len(self.list_img_files), 0)

        video2img(self.video_filename, self.raw_img_folder, frequency=2)

        self.assertGreaterEqual(len(self.list_img_files), 1)


# class TestVideoNetwork(TestVideoLocal):
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.config = Config('M:/ma/graph-training/data/short_video.mp4')
#         cls.raw_img_folder = cls.config.raw_img_folder
#         cls.video_filename = cls.config.filepath
#         cls.list_of_folders = cls.config.list_of_folders


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
