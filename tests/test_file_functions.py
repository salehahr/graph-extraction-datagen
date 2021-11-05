import os
import unittest

from config import Config
from functions.files import remove_data_folders, make_folders, clone_data_folders

base_path = '/graphics/scratch/schuelej/sar/graph-training/data'


class TestFileFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        video_fp = os.path.join(base_path, 'test/short_video.mp4')
        cls.temp_folder = os.path.join(base_path, 'test/temp')

        cls.config = Config(video_fp, trim_times=None)

        if os.path.isdir(cls.config.basename):
            clone_data_folders(cls.config.basename, cls.temp_folder)
            remove_data_folders(cls.config.basename)

    def test_make_folders(self):
        self.assertFalse(os.path.isdir(self.config.basename))
        self.assertTrue(os.path.isdir(self.temp_folder))

        self.assertFalse(os.path.isdir(self.config.raw_img_folder))
        self.assertFalse(os.path.isdir(self.config.overlay_img_folder))
        make_folders(self.config)
        self.assertTrue(os.path.isdir(self.config.raw_img_folder))
        self.assertTrue(os.path.isdir(self.config.overlay_img_folder))

    @classmethod
    def tearDownClass(cls) -> None:
        """ Restore data. """
        remove_data_folders(cls.config.basename)
        clone_data_folders(cls.temp_folder, cls.config.basename)
        remove_data_folders(cls.temp_folder)
