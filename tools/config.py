import glob
import os
import re

from tools.videos import convert_to_mp4, generate_time_tag_from_interval, trim_video

# Time tag pattern
ttag_pattern = "(.*)_(\d{4}_\d{5}__\d{4}_\d{5})"

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

node_pos_save = False
node_pos_img_save = False
adj_matr_save = False
graph_save = True


class Config:
    def __init__(
        self,
        filepath: str,
        frequency: int,
        img_length: int,
        trim_times: list,
        do_trim: bool = True,
        start=None,
        end=None,
        synthetic: bool = False,
        use_images: bool = False,
    ):
        self.use_images = use_images

        # convert non mp4 files to mp4
        if not use_images:
            ext = os.path.splitext(filepath)[1]
            if "mp4" not in ext.lower():
                filepath = convert_to_mp4(filepath)
                self.ext = ".mp4"
            else:
                self.ext = ext

        self.filepath = os.path.abspath(filepath)
        self.img_length = img_length
        self.is_synthetic = synthetic
        if not self.use_images:
            self.trim_times = trim_times
            self.sections = [self]
            self.frequency = frequency
            self._start = start
            self._end = end


        # trim properties
        if not use_images:
            ttag_match = re.search(ttag_pattern, self.filepath)
            self.is_trimmed = True if ttag_match is not None else False
            self.has_trimmed = True if self.trim_times else False

        # base folder
        if not use_images:
            self.name, self.ttag = self._get_video_name_ttag()
            self.base_folder = self._generate_base_folder_name()
            if not os.path.isdir(self.base_folder):
                os.makedirs(self.base_folder)
        else:
            self.base_folder = self.filepath

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
        if not use_images:
            if self.has_trimmed:
                if do_trim:
                    section_filepaths = trim_video(self)
                else:
                    section_filepaths = [
                        os.path.join(
                            self.base_folder,
                            f"{self.name}_" + generate_time_tag_from_interval(i) + self.ext,
                        )
                        for i in trim_times
                    ]
                self.sections = [
                    Config(
                        fp,
                        frequency=self.frequency,
                        img_length=self.img_length,
                        trim_times=[],
                        start=trim_times[i][0],
                        end=trim_times[i][1],
                    )
                    for i, fp in enumerate(section_filepaths)
                ]
            else:
                self._generate_folder_paths()

            self._generate_start_time(start)

    def _generate_folder_paths(self):
        # TODO: raw images on outside of directory
        self.raw_img_folder = f"{self.base_folder}/raw"
        self.cropped_img_folder = f"{self.base_folder}/cropped"
        self.filtered_img_folder = f"{self.base_folder}/filtered"
        self.masked_img_folder = f"{self.base_folder}/masked"
        self.threshed_img_folder = f"{self.base_folder}/threshed"
        self.skeleton_img_folder = f"{self.base_folder}/skeleton"
        self.landmarks_img_folder = f"{self.base_folder}/landmarks"
        self.node_positions_folder = f"{self.base_folder}/node_positions"
        self.adj_matr_folder = f"{self.base_folder}/adj_matr"
        self.graph_folder = f"{self.base_folder}/graphs"
        self.poly_graph_img_folder = f"{self.base_folder}/poly_graph"
        self.overlay_img_folder = f"{self.base_folder}/overlay"

    def _generate_start_time(self, start):
        if self.is_trimmed:
            if start is None:
                start_pattern = "(\d{4})_(\d{5})__\d{4}_\d{5}\."
                match = re.search(start_pattern, self.filepath)

                minutes = int(match.group(1))
                milliseconds = int(match.group(2))

                self._start = minutes * 60 + milliseconds / 1000
            else:
                self._start = start
        else:
            self._start = 0
        assert self._start is not None

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

    def _get_video_name_ttag(self):
        base_name = os.path.splitext(os.path.basename(self.filepath))[0]
        if self.is_trimmed:
            match = re.search(ttag_pattern, base_name)
            return match.group(1), match.group(2)
        else:
            return base_name, None

    def _generate_base_folder_name(self):
        directory = os.path.dirname(self.filepath)
        img_length = str(self.img_length)

        if self.is_trimmed:
            if img_length in directory and self.name in directory:
                return os.path.join(directory, self.ttag)
            else:
                return os.path.join(directory, img_length, self.name, self.ttag)

        else:
            return os.path.join(directory, img_length, self.name)

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
                self.skeleton_img_folder,
                self.landmarks_img_folder,
                self.poly_graph_img_folder,
                self.overlay_img_folder,
                self.node_positions_folder,
                self.adj_matr_folder,
                self.graph_folder,
            ]

    @property
    def raw_image_files(self):
        if self.use_images:
            dir_list = []

            for root, dirs, files in os.walk(self.filepath):
                for file in files:
                    dir_list.append(os.path.join(root, file))
        else:
            dir_list = glob.glob(os.path.join(self.base_folder, "**/raw/*.png"), recursive=True)
        return dir_list

    @property
    def cropped_image_files(self):
        if self.use_images:
            dir_list = []

            for root, dirs, files in os.walk(self.filepath):
                for file in files:
                    dir_list.append(os.path.join(root, file))
        else:
            dir_list = glob.glob(os.path.join(self.base_folder, "**/cropped/*.png"), recursive=True)
        return dir_list

    @property
    def filtered_image_files(self):
        if self.use_images:
            dir_list = []

            for root, dirs, files in os.walk(self.filepath + "\\filtered"):
                for file in files:
                    dir_list.append(os.path.join(root, file))
        else:
            dir_list = glob.glob(os.path.join(self.base_folder, "**/filtered/*.png"), recursive=True)
        return dir_list

    @property
    def masked_image_files(self):
        if self.use_images:
            dir_list = []

            for root, dirs, files in os.walk(self.filepath + "\\masked"):
                for file in files:
                    dir_list.append(os.path.join(root, file))
        else:
            dir_list = glob.glob(os.path.join(self.base_folder, "**/masked/*.png"), recursive=True)
        return dir_list

    @property
    def threshed_image_files(self):
        if self.use_images:
            dir_list = []

            for root, dirs, files in os.walk(self.filepath + "\\threshed"):
                for file in files:
                    dir_list.append(os.path.join(root, file))
        else:
            dir_list = glob.glob(os.path.join(self.base_folder, "**/threshed/*.png"), recursive=True)
        return dir_list

    @property
    def skeletonised_image_files(self):
        if self.use_images:
            dir_list = []

            for root, dirs, files in os.walk(self.filepath + "\\skeleton"):
                for file in files:
                    dir_list.append(os.path.join(root, file))
        else:
            dir_list = glob.glob(os.path.join(self.base_folder, "**/skeleton/*.png"), recursive=True)
        return dir_list

    @property
    def node_position_files(self):
        if self.use_images:
            dir_list = []

            for root, dirs, files in os.walk(self.filepath):
                for file in files:
                    dir_list.append(os.path.join(root, file))
        else:
            dir_list = glob.glob(os.path.join(self.base_folder, "**/node_positions/*.npy"), recursive=True)
        return dir_list

    @property
    def node_position_img_files(self):
        if self.use_images:
            dir_list = []

            for root, dirs, files in os.walk(self.filepath):
                for file in files:
                    dir_list.append(os.path.join(root, file))
        else:
            dir_list = glob.glob(os.path.join(self.base_folder, "**/node_positions/*.png"), recursive=True)
        return dir_list

    @property
    def adj_matrix_files(self):
        return glob.glob(
            os.path.join(self.base_folder, "**/adj_matr/*.npy"), recursive=True
        )
