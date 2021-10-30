import unittest
from test_file_functions import *

import os
import glob

import config as cl
import config_network as cn

from context import delete_files, make_folders
from context import video2img


class TestVideoLocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_img_folder = cl.raw_img_folder
        cls.video_filename = cl.VIDEO_FULL_FILEPATH_EXT
        cls.list_of_folders = cl.list_of_folders

    @property
    def list_img_files(self):
        return glob.glob(os.path.join(self.raw_img_folder, '*.png'))

    def test_video2img(self):
        make_folders(self.list_of_folders)
        delete_files(self.list_img_files)
        self.assertEqual(len(self.list_img_files), 0)

        assert('short_video' in self.video_filename)
        print(self.video_filename)
        print(self.raw_img_folder)
        video2img(self.video_filename, self.raw_img_folder, frequency=2)

        self.assertGreaterEqual(len(self.list_img_files), 1)


class TestVideoNetwork(TestVideoLocal):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_img_folder = cn.raw_img_folder
        cls.video_filename = cn.VIDEO_FULL_FILEPATH_EXT
        cls.list_of_folders = cn.list_of_folders


if __name__ == '__main__':
    unittest.main()
