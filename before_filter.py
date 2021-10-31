# * Extracts raw images from video
# * Crops the video stills

from functions_files import make_folders
from functions_images import crop_imgs
from functions_videos import video2img

from config import Config


if __name__ == '__main__':
    config = Config()
    make_folders(config)
    video2img(config)
    crop_imgs(config)
