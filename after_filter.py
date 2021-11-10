import numpy as np

from functions.images import threshold_imgs, extract_graphs, apply_img_mask

from config import Config
from video_data import video_filepath, frequency, trim_times

import warnings

warnings.simplefilter('ignore', np.RankWarning)


def after_filter(conf, skip_existing=True):
    apply_img_mask(conf)
    threshold_imgs(conf)
    extract_graphs(conf, skip_existing)


if __name__ == '__main__':
    conf = Config(video_filepath, frequency, trim_times)
    after_filter(conf)
