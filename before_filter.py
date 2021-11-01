# * Extracts raw images from video
# * Crops the video stills

from functions.files import make_folders
from functions.images import crop_imgs
from functions.videos import video2img

from config import Config


if __name__ == '__main__':
    config = Config()

    for section in config.sections:
        make_folders(section)
        video2img(section)
        crop_imgs(section)
