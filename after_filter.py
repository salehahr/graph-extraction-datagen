import numpy as np

from tools.images import apply_img_mask, threshold_imgs, skeletonise_imgs
from tools.im2graph import extract_graphs

from config import Config, image_length
from video_data import video_filepath, frequency, trim_times

import warnings

warnings.simplefilter('ignore', np.RankWarning)


def after_filter(conf: Config, skip_existing: bool) -> None:
    """
    Applies image processing functions to filtered images in the video directory.
    :param conf: video-dependent configuration
    :param skip_existing: False to overwrite existing graph file.
    """
    apply_img_mask(conf)
    threshold_imgs(conf)
    skeletonise_imgs(conf)
    extract_graphs(conf, skip_existing)


if __name__ == '__main__':
    print(f'Generating {image_length}px data for\n',
          f'\t{video_filepath}')
    conf = Config(video_filepath, frequency,
                  img_length=image_length, trim_times=trim_times)
    after_filter(conf, skip_existing=True)
