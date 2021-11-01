import glob
import os
import re

from functions.videos import trim_video, generate_time_tag_from_interval

# Video
VIDEO_FULL_FILEPATH_EXT = 'data/GRK021_test.mp4'
trim_times_in_s = None
# trim_times_in_s = [[0, 1], [2, 3]]

# Plot/Save options
thr_plot = False
pr_plot = False
lm_plot = False
poly_plot = False
overlay_plot = False

thr_save = True
pr_save = True
lm_save = True
poly_save = True
overlay_save = True

# Image Dimensions
# vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

crop_top, crop_bottom = 4, 1080
crop_left, crop_right = 416, 1532

crop_height = crop_bottom - crop_top
crop_width = crop_right - crop_left

# Time tag pattern
pattern = '(.*)_(\d{4}_\d{5}__\d{4}_\d{5})\.'


class Config:
    def __init__(self,
                 filepath: str = VIDEO_FULL_FILEPATH_EXT,
                 trim_times: list = trim_times_in_s,
                 do_trim: bool = True):

        self._filepath = filepath
        self.ext = os.path.splitext(filepath)[1]
        self.trim_times = trim_times
        self.sections = [self]

        # trim video if trim_times given, else
        if self.has_trimmed:
            section_filepaths = trim_video(self) if do_trim \
                else [self.basename + '_' + generate_time_tag_from_interval(i) \
                      + self.ext for i in trim_times]
            self.sections = [Config(fp, trim_times=[]) for fp in section_filepaths]
        else:
            self._generate_folders()

    def _generate_folders(self):
        self.raw_img_folder = f'{self.basename}/raw'
        self.cropped_img_folder = f'{self.basename}/cropped'
        self.filtered_img_folder = f'{self.basename}/filtered'
        self.masked_img_folder = f'{self.basename}/masked'
        self.threshed_img_folder = f'{self.basename}/threshed'
        self.preproc_img_folder = f'{self.basename}/skeleton'
        self.landmarks_img_folder = f'{self.basename}/landmarks'
        self.poly_graph_img_folder = f'{self.basename}/poly_graph'
        self.overlay_img_folder = f'{self.basename}/overlay'

    @property
    def filepath(self):
        return self._filepath

    @filepath.setter
    def filepath(self, new_fp: str):
        assert (os.path.isfile(new_fp))
        self._filepath = new_fp
        self._generate_folders()

    @property
    def basename(self):
        if self.is_trimmed:
            match = re.search(pattern, self.filepath)
            name_without_timetags = match.group(1)
            timetags = match.group(2)

            return os.path.join(name_without_timetags, timetags)

        else:
            return os.path.splitext(self.filepath)[0]

    @property
    def list_of_folders(self):
        if self.has_trimmed:
            return None
        else:
            return [
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

    @property
    def has_trimmed(self):
        return True if self.trim_times else False

    @property
    def raw_image_files(self):
        return glob.glob(os.path.join(self.basename, '**/raw/*.png'), recursive=True)

    @property
    def cropped_image_files(self):
        return glob.glob(os.path.join(self.basename, '**/cropped/*.png'), recursive=True)

    @property
    def filtered_image_files(self):
        return glob.glob(os.path.join(self.basename, '**/filtered/*.png'), recursive=True)

    @property
    def masked_image_files(self):
        return glob.glob(os.path.join(self.basename, '**/masked/*.png'), recursive=True)
