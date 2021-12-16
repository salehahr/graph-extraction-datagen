import glob
import os
import re

from tools.videos import trim_video, generate_time_tag_from_interval

# Time tag pattern
pattern = '(.*)_(\d{4}_\d{5}__\d{4}_\d{5})'

# Image and mask dimensions
image_length = 256
image_centre = (int(image_length / 2), int(image_length / 2))
mask_radius = 102.5 if image_length == 256 else 205

# Border attributes for classifying nodes
border_size = 2
border_radius = int(mask_radius - border_size)

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

node_pos_save = True
node_pos_img_save = True
adj_matr_save = True
graph_save = True


class Config:
    def __init__(self,
                 filepath: str,
                 frequency: int,
                 img_length: int,
                 trim_times: list,
                 do_trim: bool = True,
                 start=None,
                 end=None,
                 synthetic: bool = False):

        self.filepath = filepath
        self.ext = os.path.splitext(filepath)[1]
        self.trim_times = trim_times
        self.sections = [self]
        self.frequency = frequency
        self.img_length = img_length

        self.is_synthetic = synthetic

        self._start = start
        self._end = end

        # plot/save options
        self.thr_plot = thr_plot
        self.pr_plot = pr_plot
        self.lm_plot = lm_plot
        self.poly_plot = poly_plot
        self.overlay_plot = overlay_plot

        self.thr_save = thr_save
        self.pr_save = pr_save
        self.lm_save = lm_save
        self.poly_save = poly_save
        self.overlay_save = overlay_save

        self.node_pos_save = node_pos_save
        self.node_pos_img_save = node_pos_img_save
        self.adj_matr_save = adj_matr_save
        self.graph_save = graph_save

        # trim video if trim_times given, else
        if self.has_trimmed:
            if do_trim:
                section_filepaths = trim_video(self)
            else:
                section_filepaths = [self.basename + '_' + generate_time_tag_from_interval(i)
                                     + self.ext for i in trim_times]
            self.sections = [Config(fp,
                                    frequency=self.frequency,
                                    img_length=self.img_length,
                                    trim_times=[],
                                    start=trim_times[i][0],
                                    end=trim_times[i][1])
                             for i, fp in enumerate(section_filepaths)]
        else:
            self._generate_folders()

        self._generate_start_time(start)

    def _generate_folders(self):
        # TODO: raw images on outside of directory
        self.raw_img_folder = f'{self.basename}/raw'
        self.cropped_img_folder = f'{self.basename}/cropped'
        self.filtered_img_folder = f'{self.basename}/filtered'
        self.masked_img_folder = f'{self.basename}/masked'
        self.threshed_img_folder = f'{self.basename}/threshed'
        self.preproc_img_folder = f'{self.basename}/skeleton'
        self.landmarks_img_folder = f'{self.basename}/landmarks'
        self.node_positions_folder = f'{self.basename}/node_positions'
        self.adj_matr_folder = f'{self.basename}/adj_matr'
        self.graph_folder = f'{self.basename}/graphs'
        self.poly_graph_img_folder = f'{self.basename}/poly_graph'
        self.overlay_img_folder = f'{self.basename}/overlay'

    def _generate_start_time(self, start):
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

    def save_all(self):
        self.thr_save = True
        self.pr_save = True
        self.lm_save = True
        self.poly_save = True
        self.overlay_save = True

        self.node_pos_save = True
        self.node_pos_img_save = True
        self.adj_matr_save = True
        self.graph_save = True

    @property
    def basename(self):
        directory = os.path.dirname(self.filepath)
        img_length = str(self.img_length)
        base_name = os.path.splitext(os.path.basename(self.filepath))[0]

        if self.is_trimmed:
            match = re.search(pattern, base_name)
            name_without_timetags = match.group(1)
            timetags = match.group(2)

            return os.path.join(directory, img_length, name_without_timetags, timetags)

        else:
            return os.path.join(directory, img_length, base_name)

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
                self.node_positions_folder,
                self.adj_matr_folder,
                self.graph_folder,
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

    @property
    def threshed_image_files(self):
        return glob.glob(os.path.join(self.basename, '**/threshed/*.png'), recursive=True)

    @property
    def skeletonised_image_files(self):
        return glob.glob(os.path.join(self.basename, '**/skeleton/*.png'), recursive=True)

    @property
    def node_position_files(self):
        return glob.glob(os.path.join(self.basename, '**/node_positions/*.npy'), recursive=True)

    @property
    def node_position_img_files(self):
        return glob.glob(os.path.join(self.basename, '**/node_positions/*.png'), recursive=True)

    @property
    def adj_matrix_files(self):
        return glob.glob(os.path.join(self.basename, '**/adj_matr/*.npy'), recursive=True)
