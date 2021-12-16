# * Extracts raw images from video
# * Crops the video stills

from tools.files import make_folders
from tools.images import crop_imgs
from tools.videos import video2img

from config import Config, image_length
from video_data import video_filepath, frequency, trim_times, is_synthetic


def before_filter(conf=None):
    for section in conf.sections:
        make_folders(section)
        video2img(section)

    crop_imgs(conf)

    assert(len(conf.raw_image_files) == len(conf.cropped_image_files))


if __name__ == '__main__':
    print(f'Generating {image_length}px data for\n',
          f'\t{video_filepath}')
    conf = Config(video_filepath, frequency,
                  img_length=image_length, trim_times=trim_times,
                  synthetic=is_synthetic)
    before_filter(conf)
