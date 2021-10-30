import os
import glob
import cv2

from config import *


def is_cropped(img):
    h, w, _ = img.shape
    return (h == crop_height) and (w == crop_width)


def crop_imgs(raw_img_folder, cropped_img_folder):
    raw_img_dir = os.path.join(os.getcwd(), raw_img_folder)
    cropped_img_dir = os.path.join(os.getcwd(), cropped_img_folder)

    filepaths = glob.glob(raw_img_dir + '/*')

    for fp in filepaths:
        img = cv2.imread(fp, cv2.IMREAD_COLOR)

        if is_cropped(img):
            continue

        img_cropped = img[crop_top:crop_bottom, crop_left:crop_right]
        assert (is_cropped(img_cropped))

        # cv2.imshow('title', img_cropped)
        # cv2.waitKey()

        new_fp = os.path.join(cropped_img_dir, os.path.basename(fp))
        cv2.imwrite(new_fp, img_cropped)
