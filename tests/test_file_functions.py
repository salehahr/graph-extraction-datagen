import os
import unittest

from tools.config import Config
from tools.files import clone_data_folders, make_folders, remove_data_folders

base_path = os.path.join(os.getcwd(), "../data/test")


class TestFileFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        video_fp = os.path.join(base_path, "short_video.mp4")
        cls.temp_folder = os.path.join(base_path, "temp")
        if os.path.isdir(cls.temp_folder):
            remove_data_folders(cls.temp_folder)

        cls.config = Config(video_fp, frequency=2, img_length=512, trim_times=None)
        cls.originals_exist = True if os.path.isdir(cls.config.base_folder) else False

        if cls.originals_exist:
            clone_data_folders(cls.config.base_folder, cls.temp_folder)
            remove_data_folders(cls.config.base_folder)

    def test_make_folders(self):
        self.assertFalse(os.path.isdir(self.config.base_folder))
        if self.originals_exist:
            self.assertTrue(os.path.isdir(self.temp_folder))

        self.assertFalse(os.path.isdir(self.config.raw_img_folder))
        self.assertFalse(os.path.isdir(self.config.overlay_img_folder))
        make_folders(self.config)
        self.assertTrue(os.path.isdir(self.config.raw_img_folder))
        self.assertTrue(os.path.isdir(self.config.overlay_img_folder))

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore data if originals present."""
        if cls.originals_exist:
            remove_data_folders(cls.config.base_folder)
            clone_data_folders(cls.temp_folder, cls.config.base_folder)
            remove_data_folders(cls.temp_folder)
