import os
import shutil
import unittest

from after_filter import after_filter
from before_filter import before_filter
from tools.config import Config
from tools.files import delete_files, make_folders
from tools.videos import make_video_clip, trim_video

data_path = os.path.join(os.getcwd(), "../data/test")


class TestVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = None

    def test_before_filter(self):
        if self.config:
            self.config.save_all()

            delete_files(self.config.raw_image_files)
            delete_files(self.config.cropped_image_files)
            self.assertEqual(len(self.config.raw_image_files), 0)
            self.assertEqual(len(self.config.cropped_image_files), 0)

            before_filter(self.config)

            self.assertGreaterEqual(len(self.config.raw_image_files), 1)
            self.assertGreaterEqual(len(self.config.cropped_image_files), 1)

    def test_after_filter(self):
        if self.config:
            self.config.overlay_plot = True
            self.config.lm_plot = True

            delete_files(self.config.masked_image_files)

            self.assertGreaterEqual(len(self.config.filtered_image_files), 1)
            self.assertEqual(len(self.config.masked_image_files), 0)

            after_filter(self.config, skip_existing=False)

            self.assertGreaterEqual(len(self.config.masked_image_files), 1)
            self.assertGreaterEqual(len(self.config.adj_matrix_files), 1)

            self.assertEqual(
                len(self.config.masked_image_files),
                len(self.config.node_position_files),
            )
            self.assertEqual(
                len(self.config.node_position_img_files),
                len(self.config.node_position_files),
            )


class TestShortVideo(TestVideo):
    @classmethod
    def setUpClass(cls) -> None:
        video_fp = os.path.join(data_path, "short_video.mp4")
        cls.config = Config(video_fp, frequency=2, img_length=512, trim_times=[])

    def test_is_not_trimmed(self):
        self.assertFalse(self.config.is_trimmed)


class TestTrimmedVideo(TestVideo):
    @classmethod
    def setUpClass(cls) -> None:
        video_fp = os.path.join(
            data_path, "512/trimmed/trimmed_0000_02000__0000_03000.mp4"
        )
        cls.config = Config(video_fp, frequency=2, img_length=512, trim_times=[])

    def test_is_trimmed(self):
        self.assertTrue(self.config.is_trimmed)


class TestMultiSectionVideo(TestVideo):
    @classmethod
    def setUpClass(cls) -> None:
        video_fp = os.path.join(data_path, "trimmed.mp4")
        trim_times = [[2, 3], [4, 5]]
        cls.config = Config(
            video_fp, frequency=2, img_length=512, trim_times=trim_times
        )

    def test_has_sections(self):
        self.assertIsNotNone(self.config.sections)
        self.assertTrue(self.config.has_trimmed)
        self.assertFalse(self.config.is_trimmed)

    def test_make_folders(self):
        for section in self.config.sections:
            make_folders(section)


class TestBuggyVideo(TestVideo):
    """Video for which the final frame keeps getting extracted
    in a continous loop."""

    @classmethod
    def setUpClass(cls) -> None:
        video_fp = os.path.join(data_path, "synthetic-bladder11.mp4")
        cls.config = Config(video_fp, frequency=2, img_length=256, trim_times=[])

    def test_after_filter(self):
        pass


@unittest.skip("Skip trimming video")
class TestTrimVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.orig_filename = os.path.join(data_path, "GRK021_test.mp4")
        trim_times = [[2, 3]]
        cls.config = Config(
            cls.orig_filename,
            frequency=2,
            img_length=512,
            trim_times=trim_times,
            do_trim=False,
        )

    def test_trim_video_with_target(self):
        target_filename = os.path.join(
            data_path, "512/trimmed/trimmed_0000_02000__0000_03000.mp4"
        )
        self.assertTrue(os.path.isfile(self.orig_filename))
        try:
            os.remove(target_filename)
        except FileNotFoundError:
            pass

        trim_video(self.config, [target_filename])

        self.assertTrue(os.path.isfile(target_filename))

    def test_trim_video_without_target(self):
        self.assertTrue(os.path.isfile(self.orig_filename))
        trimmed_fp = os.path.join(
            self.config.base_folder, f"{self.config.name}_0000_02000__0000_03000.mp4"
        )

        os.makedirs(self.config.base_folder)

        trim_video(self.config)
        self.assertTrue(os.path.isfile(trimmed_fp))

        shutil.rmtree(self.config.base_folder)


@unittest.skip("Skip trimming video")
class TestTrimVideoSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.orig_filename = os.path.join(data_path, "GRK021_test.mp4")
        trim_times = [[2, 3], [4, 5]]
        cls.config = Config(
            cls.orig_filename,
            frequency=2,
            img_length=512,
            trim_times=trim_times,
            do_trim=False,
        )

    def test_trim_video_with_target(self):
        self.assertTrue(os.path.isfile(self.orig_filename))

        target_filename_1 = os.path.join(
            data_path, "512/trimmed/trimmed_0000_02000__0000_03000.mp4"
        )
        target_filename_2 = os.path.join(
            data_path, "512/trimmed/trimmed_0000_04000__0000_05000.mp4"
        )
        target_filenames = [target_filename_1, target_filename_2]

        for fn in target_filenames:
            if os.path.isfile(fn):
                os.remove(fn)

        trim_video(self.config, target_filenames)

        for fn in target_filenames:
            self.assertTrue(os.path.isfile(fn))

    def test_trim_video_without_target(self):
        self.assertTrue(os.path.isfile(self.orig_filename))

        os.makedirs(self.config.base_folder)

        sections = trim_video(self.config)

        for fp in sections:
            self.assertTrue(os.path.isfile(fp))
            os.remove(fp)

        shutil.rmtree(self.config.base_folder)


class TestMakeVideoClip(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_make_video_clip(self):
        source_path = os.path.join(data_path, "filtered")
        target_fp = os.path.join(data_path, "filtered.mp4")

        if os.path.isfile(target_fp):
            os.remove(target_fp)

        self.assertTrue(os.path.isdir(source_path))

        make_video_clip(source_path, target_fp, fps=25)

        self.assertTrue(os.path.isfile(target_fp))
