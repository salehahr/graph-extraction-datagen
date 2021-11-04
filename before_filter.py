# * Extracts raw images from video
# * Crops the video stills

from functions.files import make_folders
from functions.images import crop_imgs
from functions.videos import video2img

from config import Config


def before_filter(conf=None):
    for section in conf.sections:
        make_folders(section)
        video2img(section)

    crop_imgs(conf)

    assert(len(conf.raw_image_files) == len(conf.cropped_image_files))


if __name__ == '__main__':
    conf = Config()
    before_filter(conf)