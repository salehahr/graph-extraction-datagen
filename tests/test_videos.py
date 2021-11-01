import glob
import os
import unittest
import context

from functions.files import make_folders, delete_files
from functions.videos import video2img, trim_video
from functions.images import crop_imgs

from config import Config


class TestVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = None
        cls.raw_img_folder = None

    @property
    def list_img_files(self):
        if not self.config.has_trimmed:
            return glob.glob(os.path.join(self.raw_img_folder, '*.png'))
        else:
            list_imgs = []
            for section in self.config.sections:
                list_imgs += glob.glob(os.path.join(section.raw_img_folder, '*.png'))
            return list_imgs

    def before_filter(self):
        for section in self.config.sections:
            make_folders(section)
            video2img(section, frequency=2)
            crop_imgs(section)

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


class TestMultiSectionVideo(TestVideo):
    @classmethod
    def setUpClass(cls) -> None:
        filename = "trimmed.mp4"
        trim_times = [[2, 3], [4, 5]]

        cls.config = Config(filename, trim_times)

    def test_has_sections(self):
        self.assertIsNotNone(self.config.sections)
        self.assertTrue(self.config.has_trimmed)
        self.assertFalse(self.config.is_trimmed)

    def test_make_folders(self):
        for section in self.config.sections:
            make_folders(section)


# class TestVideoNetwork(TestVideo):
#     @classmethod
#     def setUpClass(cls) -> None:
#         cls.config = Config('M:/ma/graph-training/data/short_video.mp4')
#         cls.raw_img_folder = cls.config.raw_img_folder


class TestTrimVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.orig_filename = "../data/GRK021_test.mp4"
        trim_times = [[2, 3]]

        cls.config = Config(cls.orig_filename, trim_times)

        cls.target_filename = "trimmed_0000_02000__0000_03000.mp4"

    def test_trim_video(self):
        self.assertTrue(os.path.isfile(self.orig_filename))
        os.remove(self.target_filename)

        trim_video(self.config, [self.target_filename])

        self.assertTrue(os.path.isfile(self.target_filename))


class TestTrimVideoSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.orig_filename = "../data/GRK021_test.mp4"
        trim_times = [[2, 3], [4, 5]]

        cls.config = Config(cls.orig_filename, trim_times)

        cls.target_filename_1 = "trimmed_0000_02000__0000_03000.mp4"
        cls.target_filename_2 = "trimmed_0000_04000__0000_05000.mp4"
        cls.target_filenames = [cls.target_filename_1, cls.target_filename_2]

    def test_trim_video(self):
        self.assertTrue(os.path.isfile(self.orig_filename))

        for fn in self.target_filenames:
            if os.path.isfile(fn):
                os.remove(fn)

        trim_video(self.config, self.target_filenames)

        for fn in self.target_filenames:
            self.assertTrue(os.path.isfile(fn))
