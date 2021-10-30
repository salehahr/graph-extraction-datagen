# * Extracts raw images from video
# * Crops the video stills

from functions_images import crop_imgs
from video2img import video2img

from config import make_folders
from config import VIDEO_FILENAME_EXT, raw_img_folder, cropped_img_folder


if __name__ == '__main__':
    make_folders()
    video2img(VIDEO_FILENAME_EXT, raw_img_folder)
    crop_imgs(raw_img_folder, cropped_img_folder)
