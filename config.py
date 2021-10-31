import os
import re
from dataclasses import dataclass

# Video
VIDEO_FULL_FILEPATH_EXT = 'data/GRK021_test.mp4'

# Image Dimensions
# vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

crop_top, crop_bottom = 4, 1080
crop_left, crop_right = 416, 1532

crop_height = crop_bottom - crop_top
crop_width = crop_right - crop_left

# Time tag pattern
pattern = '(.*)_(\d{4}_\d{5}__\d{4}_\d{5})\.'


@dataclass
class Config:
    _filepath: str = VIDEO_FULL_FILEPATH_EXT

    def __post_init__(self):
        self.generate_folders()

    @property
    def filepath(self):
        return self._filepath

    @filepath.setter
    def filepath(self, new_fp: str):
        assert (os.path.isfile(new_fp))
        self._filepath = new_fp
        self.generate_folders()

    def get_basename(self):
        if self.is_trimmed:
            match = re.search(pattern, self.filepath)
            name_without_timetags = match.group(1)
            timetags = match.group(2)

            return os.path.join(name_without_timetags, timetags)

        else:
            return os.path.splitext(self.filepath)[0]

    def generate_folders(self):
        basename = self.get_basename()

        self.raw_img_folder = f'{basename}/raw'
        self.cropped_img_folder = f'{basename}/cropped'
        self.filtered_img_folder = f'{basename}/filtered'
        self.masked_img_folder = f'{basename}/masked'
        self.threshed_img_folder = f'{basename}/threshed'
        self.preproc_img_folder = f'{basename}/skeleton'
        self.landmarks_img_folder = f'{basename}/landmarks'
        self.poly_graph_img_folder = f'{basename}/poly_graph'
        self.overlay_img_folder = f'{basename}/overlay'

        self.list_of_folders = [
            self.raw_img_folder,
            self.cropped_img_folder,
            self.filtered_img_folder,
            self.masked_img_folder,
            self.threshed_img_folder,
            self.preproc_img_folder,
            self.landmarks_img_folder,
            self.poly_graph_img_folder,
            self.overlay_img_folder,
        ]

    @property
    def is_trimmed(self):
        match = re.search(pattern, self.filepath)
        return True if match is not None else False
