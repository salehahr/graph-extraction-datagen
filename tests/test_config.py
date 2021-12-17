import unittest

from config import GeneralConfig, ImageConfig, VideoConfig


class TestVideoConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = VideoConfig("videos.yaml", "untrimmed")

    def test_absolute_video_path(self):
        fp = "/graphics/scratch/schuelej/sar/graph-training/tests/synthetic-bladder11.mp4"
        self.assertEqual(self.config.filepath, fp)


class TestImageConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ImageConfig("config.yaml")

    def test_config(self):
        self.assertIsNotNone(self.config)


class TestGeneralConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = GeneralConfig("config.yaml", "videos.yaml", "synthetic-bladder11")

    def test_config(self):
        self.assertIsNotNone(self.config)
        return
