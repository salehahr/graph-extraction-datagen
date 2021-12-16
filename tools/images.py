import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from skimage import morphology
from skimage.morphology import skeletonize

from config import image_centre, border_size, border_radius


blur_kernel = (5, 5)


def get_rgb(img):
    """ Gets RGB image for matplotlib plots. """
    n_channels = img.shape[2] if len(img.shape) >= 3 else 1

    if n_channels == 1:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_centre(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_thresholded():
        # create background image
        bg = cv2.dilate(img_grey, np.ones(blur_kernel, dtype=np.uint8))
        bg = cv2.GaussianBlur(bg, blur_kernel, 1)

        # subtract out background from source
        src_no_bg = 255 - cv2.absdiff(img_grey, bg)

        # threshold
        _, thresh = cv2.threshold(src_no_bg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return thresh

    # calculate moments of binary image
    moments = cv2.moments(get_thresholded())

    # calculate x,y coordinate of center
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def crop_imgs(conf):
    for fp in conf.raw_image_files:
        new_fp = fp.replace('raw', 'cropped')

        # uncomment to skip existing files
        # if os.path.isfile(new_fp):
        #     continue

        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        img_cropped = crop_resize_square(img, conf.img_length, conf.is_synthetic)

        cv2.imwrite(new_fp, img_cropped)


def crop_resize_square(img: np.ndarray, length: int, is_synthetic: bool):
    return resize_square(centre_crop(img, is_synthetic), length)


def resize_square(img: np.ndarray, length: int):
    return cv2.resize(img, (length, length))


def centre_crop(img: np.ndarray, is_synthetic):
    """
    Extracts a square from the original image,
    from around the original image's centroid.
    """
    min_y, min_x = 0, 0
    max_y, max_x, _ = img.shape

    if not is_synthetic:
        cx, cy = get_centre(img)
        crop_radius = 575
    else:
        cx = int(img.shape[1] / 2)
        cy = int(img.shape[0] / 2)
        crop_radius = int(img.shape[0] / 2)

    # get crop dimensions
    crop_left = cx - crop_radius
    crop_right = cx + crop_radius
    crop_top = cy - crop_radius
    crop_bottom = cy + crop_radius

    # calculate padding
    top_pad = 0 if crop_top >= 0 else -crop_top
    bottom_pad = 0 if crop_bottom <= max_y else crop_bottom - max_y
    left_pad = 0 if crop_left >= 0 else -crop_left
    right_pad = 0 if crop_right <= max_x else crop_right - max_x

    # set crop dimensions to max edges if necessary
    crop_left = crop_left if crop_left >= 0 else min_x
    crop_right = crop_right if crop_right <= max_x else max_x
    crop_top = crop_top if crop_top >= 0 else min_y
    crop_bottom = crop_bottom if crop_bottom <= max_y else max_y

    img = img[crop_top:crop_bottom, crop_left:crop_right]

    return cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad,
                              cv2.BORDER_CONSTANT, value=[0, 0, 0])


def apply_img_mask(conf):
    mask = create_mask(conf.img_length)

    for fp in conf.filtered_image_files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        masked = np.multiply(mask, img)

        new_fp = fp.replace('filtered', 'masked')
        cv2.imwrite(new_fp, masked)


def create_mask(img_length: int) -> np.ndarray:
    centre = (int(img_length / 2), int(img_length / 2))
    mask_radius = 102.5 if img_length == 256 else 205

    mask = np.zeros((img_length, img_length), np.float32)
    cv2.circle(mask, centre, int(mask_radius), (1., 1., 1.), -1)

    return mask


def threshold_imgs(conf):
    for fp in conf.masked_image_files:
        new_fp = fp.replace('masked', 'threshed')
        img = cv2.imread(fp, 0)
        threshold(img, conf.thr_save, new_fp)


def threshold(filtered_img: np.ndarray, do_save: bool, filepath: str = '') -> np.ndarray:
    blurred_img = cv2.GaussianBlur(filtered_img, blur_kernel, 0)
    _, thresholded_img = cv2.threshold(blurred_img, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if do_save:
        cv2.imwrite(filepath, thresholded_img)

    return thresholded_img


def skeletonise_imgs(conf):
    for fp in conf.threshed_image_files:
        skel_fp = fp.replace('threshed', 'skeleton')

        img_threshed = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)

        skeletonise_and_clean(img_threshed,
                              conf.pr_plot, conf.pr_save, skel_fp)


def skeletonise_and_clean(thr_image: np.ndarray, plot: bool, save: bool, directory: str):
    """
    Creates skeletonised image
    :param thr_image: thresholded image
    :param plot:
    :param save:
    :param directory:
    :return:
    """
    edgelength = 10

    # skeletonize
    img = thr_image.copy() / 255
    img = img.astype(int)
    skeleton_noisy = skeletonize(img).astype(int) * 255

    # remove too small edges
    bool_img = (skeleton_noisy.copy() / 255).astype(bool)
    labeled = morphology.label(bool_img)
    skeleton = morphology.remove_small_objects(labeled, edgelength + 1)
    skeleton[skeleton > 0] = 255
    skeleton = np.uint8(skeleton)

    remove_bug_pixels(skeleton)
    set_black_border(skeleton)

    if plot:
        fig, axes = plt.subplots(1, 2)
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])

        axes[0].imshow(thr_image, 'gray')
        axes[0].set_title('thresholded')

        axes[1].imshow(skeleton, 'gray')
        axes[1].set_title('skeletonised')

        plt.show()

    if save:
        cv2.imwrite(directory, skeleton)

    return np.uint8(skeleton)


def remove_bug_pixels(skeleton: np.ndarray):
    # bug pixel elimination based on
    # "Preprocessing and postprocessing for skeleton-based fingerprint minutiae extraction"
    bug_pixels = []
    for x in range(1, skeleton.shape[0] - 1):
        for y in range(1, skeleton.shape[1] - 1):
            if skeleton[x, y] == 255:
                s = num_in_4connectivity(x, y, skeleton)

                if s > 2:
                    bug_pixels.append([x, y])

    for bpx, bpy in bug_pixels:
        s = num_in_4connectivity(bpx, bpy, skeleton)

        if s > 2:
            skeleton[bpx, bpy] = 0


def set_black_border(img: np.ndarray):
    mask = np.ones(img.shape, dtype=np.int8)

    mask[:, 0] = 0
    mask[:, -1] = 0
    mask[0, :] = 0
    mask[-1, :] = 0

    np.uint8(np.multiply(mask, img))


def four_connectivity(a: int, b: int):
    # list of pixels in 4-connectivity of [a,b]
    return [[a + 1, b], [a - 1, b], [a, b + 1], [a, b - 1]]


def num_in_4connectivity(a: int, b: int, image: np.ndarray):
    # how many pixel with value 255 are in 4-connectivity of [a,b]
    neighbours = four_connectivity(a, b)

    count = 0
    for nr, nc in neighbours:
        if image[nr, nc] == 255:
            count += 1

    return count


def overlay_border(img: np.ndarray):
    """"
    Applies an overlay of the mask border to the image
    """
    bgr_yellow = (0, 255, 255)
    cv2.circle(img, image_centre, border_radius, bgr_yellow, border_size)


def generate_node_pos_img(graph, img_length):
    img = np.zeros((img_length, img_length)).astype(np.uint8)

    for x, y in graph.positions:
        row, col = y, x
        img[row][col] = 255

    return img
