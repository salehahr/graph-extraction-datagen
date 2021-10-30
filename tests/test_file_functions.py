import os
import unittest

import config as cl
from functions_files import remove_data_folders, make_folders
from tests import config_network as cn


class TestFileFunctionsLocal(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        remove_data_folders('short_video')

    def test_make_folders(self):
        self.assertFalse(os.path.isdir(cl.raw_img_folder))
        self.assertFalse(os.path.isdir(cl.overlay_img_folder))
        make_folders(cl.list_of_folders)
        self.assertTrue(os.path.isdir(cl.raw_img_folder))
        self.assertTrue(os.path.isdir(cl.overlay_img_folder))


class TestFileFunctionsNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        remove_data_folders('M:/ma/graph-training/data/short_video')

    def test_make_folders(self):
        self.assertFalse(os.path.isdir(cn.raw_img_folder))
        self.assertFalse(os.path.isdir(cn.overlay_img_folder))
        make_folders(cn.list_of_folders)
        self.assertTrue(os.path.isdir(cn.raw_img_folder))
        self.assertTrue(os.path.isdir(cn.overlay_img_folder))
