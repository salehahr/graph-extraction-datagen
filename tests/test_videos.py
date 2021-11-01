import glob
import os
import unittest
import context

from functions.files import make_folders, delete_files
from functions.videos import video2img, trim_video
from functions.images import crop_imgs, apply_img_mask

from config import Config


class TestVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = None
        cls.raw_img_folder = None

    def before_filter(self):
        for section in self.config.sections:
            make_folders(section)
            video2img(section, frequency=2)

        crop_imgs(self.config)

    def after_filter(self):
        apply_img_mask(self.config)

    def test_before_filter(self):
        if self.config:
            delete_files(self.config.raw_image_files)
            self.assertEqual(len(self.config.raw_image_files), 0)

            self.before_filter()

            self.assertGreaterEqual(len(self.config.raw_image_files), 1)

    def test_after_filter(self):
        if self.config:
            delete_files(self.config.masked_image_files)
            self.assertGreaterEqual(len(self.config.filtered_image_files), 1)
            self.assertEqual(len(self.config.masked_image_files), 0)
            self.after_filter()
            self.assertGreaterEqual(len(self.config.masked_image_files), 1)



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

    def test_trim_video_with_target(self):
        self.assertTrue(os.path.isfile(self.orig_filename))
        try:
            os.remove(self.target_filename)
        except FileNotFoundError:
            pass

        trim_video(self.config, [self.target_filename])

        self.assertTrue(os.path.isfile(self.target_filename))

    def test_trim_video_without_target(self):
        self.assertTrue(os.path.isfile(self.orig_filename))
        trimmed_fp = '../data/GRK021_test_0000_02000__0000_03000.mp4'
        try:
            os.remove(trimmed_fp)
        except FileNotFoundError:
            pass

        trim_video(self.config)

        self.assertTrue(os.path.isfile(trimmed_fp))


class TestTrimVideoSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.orig_filename = "../data/GRK021_test.mp4"
        trim_times = [[2, 3], [4, 5]]

        cls.config = Config(cls.orig_filename, trim_times, do_trim=False)

        cls.target_filename_1 = "trimmed_0000_02000__0000_03000.mp4"
        cls.target_filename_2 = "trimmed_0000_04000__0000_05000.mp4"
        cls.target_filenames = [cls.target_filename_1, cls.target_filename_2]

    def test_trim_video_with_target(self):
        self.assertTrue(os.path.isfile(self.orig_filename))

        for fn in self.target_filenames:
            if os.path.isfile(fn):
                os.remove(fn)

        trim_video(self.config, self.target_filenames)

        for fn in self.target_filenames:
            self.assertTrue(os.path.isfile(fn))

    def test_trim_video_without_target(self):
        self.assertTrue(os.path.isfile(self.orig_filename))

        sections = trim_video(self.config)

        for fp in sections:
            self.assertTrue(os.path.isfile(fp))
            os.remove(fp)
