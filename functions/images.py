import os
import glob
import cv2
import numpy as np

from config import *


if 'tests' in os.getcwd():
    mask_path = '../data/mask.png'
else:
    mask_path = 'data/mask.png'


def is_cropped(img: np.ndarray):
    h, w, _ = img.shape
    return (h == crop_height) and (w == crop_width)


def crop_imgs(config):
    for fp in config.raw_image_files:
        new_fp = fp.replace('raw', 'cropped')

        if os.path.isfile(new_fp):
            continue

        img = cv2.imread(fp, cv2.IMREAD_COLOR)

        if is_cropped(img):
            continue

        img_cropped = img[crop_top:crop_bottom, crop_left:crop_right]
        assert (is_cropped(img_cropped))

        # cv2.imshow('title', img_cropped)
        # cv2.waitKey()

        cv2.imwrite(new_fp, img_cropped)


def apply_img_mask(config):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 1  # convert non zero entries to 1

    for fp in config.filtered_image_files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        masked = np.multiply(mask, img)

        # cv2.imshow('title', masked_img)
        # cv2.waitKey()

        new_fp = fp.replace('filtered', 'masked')
        cv2.imwrite(new_fp, masked)


def thresholding(filtered_img: np.ndarray, blur_kernel: tuple,
                 do_save: bool, filepath: str = '') \
                -> np.ndarray:
    blurred_img = cv2.GaussianBlur(filtered_img, blur_kernel, 0)
    _, thresholded_img = cv2.threshold(blurred_img, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if do_save:
        cv2.imwrite(filepath, thresholded_img)

    return thresholded_img
