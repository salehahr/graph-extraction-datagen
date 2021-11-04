import glob
import os
import re

from functions.videos import trim_video, generate_time_tag_from_interval

import video_data

# Time tag pattern
pattern = '(.*)_(\d{4}_\d{5}__\d{4}_\d{5})\.'


class Config:
    def __init__(self,
                 filepath: str = video_data.VIDEO_FULL_FILEPATH_EXT,
                 trim_times: list = video_data.trim_times_in_s,
                 do_trim: bool = True,
                 start = None):

        self._filepath = filepath
        self.ext = os.path.splitext(filepath)[1]
        self.trim_times = trim_times
        self.sections = [self]
        self._start = start

        # trim video if trim_times given, else
        if self.has_trimmed:
            if do_trim:
                section_filepaths = trim_video(self)
            else:
                section_filepaths = [self.basename + '_' + generate_time_tag_from_interval(i) \
                                     + self.ext for i in trim_times]
            self.sections = [Config(fp, trim_times=[], start=trim_times[i][0]) \
                             for i, fp in enumerate(section_filepaths)]
        else:
            self._generate_folders()

        self.generate_start_time(start)

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

    def generate_start_time(self, start):
        if self.is_trimmed:
            if start is None:
                start_pattern = '(\d{4})_(\d{5})__\d{4}_\d{5}\.'
                match = re.search(start_pattern, self.filepath)

                minutes = int(match.group(1))
                milliseconds = int(match.group(2))

                self._start = minutes * 60 + milliseconds / 1000
            else:
                self._start = start
        else:
            self._start = 0
        assert (self._start is not None)

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


# Plot/Save options
thr_plot = video_data.thr_plot
pr_plot = video_data.pr_plot
lm_plot = video_data.lm_plot
poly_plot = video_data.poly_plot
overlay_plot = video_data.overlay_plot

thr_save = video_data.thr_save
pr_save = video_data.pr_save
lm_save = video_data.lm_save
poly_save = video_data.poly_save
overlay_save = video_data.overlay_save


# Image Dimensions
img_length = video_data.img_length
