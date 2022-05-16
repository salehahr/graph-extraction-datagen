import math
import os
from typing import Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import morphology
from skimage.morphology import skeletonize

from tools.config import border_radius, border_size, image_centre
from tools.Point import num_in_4connectivity

blur_kernel = (5, 5)


def normalise(img: np.ndarray) -> np.ndarray:
    img = img.copy()
    img[img == 255] = 1
    return np.uint8(img)


def get_rgb(img):
    """Gets RGB image for matplotlib plots."""
    n_channels = img.shape[2] if len(img.shape) >= 3 else 1

    if n_channels == 1:
        return img
    else:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_centre(img: np.ndarray, moments_from: str = "thresh") -> Tuple[int, int]:
    """
    Sometimes the thresholded image given out is bad.
    In that case switch to calculating moemnts from the grayscale image instead
    of the thresholded.
    :param img: BGR frame from endoscopic video
    :param moments_from: option to calculate moments from greyscale image or thresholded
    """
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def get_thresholded():
        # create background image
        bg = cv2.dilate(img_grey, np.ones(blur_kernel, dtype=np.uint8))
        bg = cv2.GaussianBlur(bg, blur_kernel, 1)

        # subtract out background from source
        src_no_bg = 255 - cv2.absdiff(img_grey, bg)

        # threshold
        _, thresh = cv2.threshold(
            src_no_bg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        return thresh

    # calculate moments of binary image
    if "thresh" in moments_from:
        moments = cv2.moments(get_thresholded())
    else:
        moments = cv2.moments(img_grey)

    # calculate x,y coordinate of center
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    return cx, cy


def crop_imgs(conf):
    if conf.use_images:
        croppath = (conf.filepath + "\\cropped").replace("\\", "/")
        if not os.path.exists(croppath):
            os.mkdir(croppath)
    for fp in conf.raw_image_files:
        if conf.use_images:
            new_fp = (
                croppath.replace("/", "\\") + "\\" + "cropped_" + fp.split("\\")[-1]
            )
            if not new_fp.endswith(".png"):
                new_fp = ".".join(new_fp.split(".")[:-1]) + ".png"
        else:
            new_fp = fp.replace("raw", "cropped")

        # uncomment to skip existing files
        # if os.path.isfile(new_fp):
        #     continue

        img = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img is not None:
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
        cx, cy = get_centre(img, moments_from="thresh")
        crop_radius = int(img.shape[0] / 2)
    else:
        cx = int(img.shape[1] / 2)
        cy = int(img.shape[0] / 2)
        crop_radius = 575

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

    return cv2.copyMakeBorder(
        img,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )


def apply_img_mask(conf):
    if conf.use_images:
        maskpath = (conf.filepath + "\\masked").replace("\\", "/")
        if not os.path.exists(maskpath):
            os.mkdir(maskpath)

    mask = create_mask(conf.img_length)

    for fp in conf.filtered_image_files:
        if conf.use_images:
            new_fp = (
                maskpath.replace("/", "\\")
                + "\\"
                + (fp.split("\\")[-1]).replace("cropped", "masked")
            )
        else:
            new_fp = fp.replace("filtered", "masked")

        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        masked = np.multiply(mask, img)

        cv2.imwrite(new_fp, masked)


def create_mask(img_length: int) -> np.ndarray:
    centre = (int(img_length / 2), int(img_length / 2))
    mask_radius = 102.5 if img_length == 256 else 205

    mask = np.zeros((img_length, img_length), np.float32)
    cv2.circle(mask, centre, int(mask_radius), (1.0, 1.0, 1.0), -1)

    return mask


def threshold_imgs(conf, bin_threshold: int = 0):
    if conf.use_images:
        threshpath = (conf.filepath + "\\threshed").replace("\\", "/")
        if not os.path.exists(threshpath):
            os.mkdir(threshpath)

    for fp in conf.masked_image_files:
        if conf.use_images:
            new_fp = (
                threshpath.replace("/", "\\")
                + "\\"
                + (fp.split("\\")[-1]).replace("masked", "threshed")
            )
        else:
            new_fp = fp.replace("masked", "threshed")
        img = cv2.imread(fp, 0)
        threshold(img, conf.thr_save, new_fp, bin_threshold)


def threshold(
    filtered_img: np.ndarray, do_save: bool, filepath: str = "", threshold: int = 0
) -> np.ndarray:
    blurred_img = cv2.GaussianBlur(filtered_img, blur_kernel, 0)
    if threshold == 0:
        thresholded_img = cv2.threshold(
            blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        thresholded_img = cv2.threshold(blurred_img, threshold, 255, cv2.THRESH_BINARY)

    if do_save:
        cv2.imwrite(filepath, thresholded_img[1])

    return thresholded_img


def skeletonise_imgs(conf):
    if conf.use_images:
        skelpath = (conf.filepath + "\\skeleton").replace("\\", "/")
        if not os.path.exists(skelpath):
            os.mkdir(skelpath)
    for fp in conf.threshed_image_files:
        if conf.use_images:
            skel_fp = (
                skelpath.replace("/", "\\")
                + "\\"
                + (fp.split("\\")[-1]).replace("threshed", "skeleton")
            )
        else:
            skel_fp = fp.replace("threshed", "skeleton")

        img_threshed = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)

        skeletonise_and_clean(img_threshed, conf.pr_plot, conf.pr_save, skel_fp)


def skeletonise_and_clean(
    thr_image: np.ndarray, plot: bool, save: bool, directory: str
):
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

        axes[0].imshow(thr_image, "gray")
        axes[0].set_title("thresholded")

        axes[1].imshow(skeleton, "gray")
        axes[1].set_title("skeletonised")

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


def overlay_border(img: np.ndarray):
    """ "
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


# By Johann
def fft_filter_vert_stripes(conf):
    tespath = conf.filepath + "/cropped"
    tes1 = os.listdir(tespath)
    tes2 = os.path.join(conf.filepath + "/cropped", tes1[1])
    files = [
        f
        for f in os.listdir(conf.filepath + "/cropped")
        if os.path.isfile(os.path.join(conf.filepath + "/cropped", f))
    ]
    for file in files:
        imgPath = os.path.join(conf.filepath + "/cropped", file)
        image = cv2.imread(imgPath)
        imgbr = cv2.split(image)
        imgbr_res = []

        for img in imgbr:
            # read input as grayscale
            # img = cv2.imread('pattern_lines.png', 0)
            hh, ww = img.shape

            # get min and max and mean values of img
            img_min = np.amin(img)
            img_max = np.amax(img)
            img_mean = int(np.mean(img))

            # pad the image to dimension a power of 2
            hhh = math.ceil(math.log2(hh))
            hhh = int(math.pow(2, hhh))
            www = math.ceil(math.log2(ww))
            www = int(math.pow(2, www))
            imgp = np.full((hhh, www), img_mean, dtype=np.uint8)
            imgp[0:hh, 0:ww] = img

            # convert image to floats and do dft saving as complex output
            dft = cv2.dft(np.float32(imgp), flags=cv2.DFT_COMPLEX_OUTPUT)

            # apply shift of origin from upper left corner to center of image
            dft_shift = np.fft.fftshift(dft)

            # extract magnitude and phase images
            mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])

            # get spectrum
            spec = np.log(mag) / 20
            min, max = np.amin(spec, (0, 1)), np.amax(spec, (0, 1))

            # threshold the spectrum to find bright spots
            thresh = (255 * spec).astype(np.uint8)
            thresh = cv2.threshold(thresh, 140, 255, cv2.THRESH_BINARY)[1]

            # cover the center columns of thresh with black
            # center_y = www // 2
            # center_x = hhh // 2
            # center = (center_y, center_x)
            # cv2.circle(thresh, center=center, radius=40, color=0, thickness=-1)
            xc = www // 2
            cv2.line(thresh, (xc, 0), (xc, hhh - 1), 0, 30)

            # get the x coordinates of the bright spots
            points = np.column_stack(np.nonzero(thresh))
            # print(points)

            # create mask from spectrum drawing vertical lines at bright spots
            mask = thresh.copy()
            # center_y = www//2
            # center_x = hhh//2
            # center = (center_y, center_x)
            for p in points:
                # x = p[0]
                y = p[1]
                # radius = round(math.sqrt((x-center_x)**2 + (y-center_y)**2))
                # cv2.circle(mask, center=center, radius=radius, color=200, thickness=2)
                cv2.line(mask, (y, 0), (y, hhh - 1), 200, 2)

            cv2.imshow("MASK", mask)
            # apply mask to magnitude such that magnitude is made black where mask is white
            mag[mask != 0] = 0

            # convert new magnitude and old phase into cartesian real and imaginary components
            real, imag = cv2.polarToCart(mag, phase)

            # combine cartesian components into one complex image
            back = cv2.merge([real, imag])

            # shift origin from center to upper left corner
            back_ishift = np.fft.ifftshift(back)

            # do idft saving as complex output
            img_back = cv2.idft(back_ishift)

            # combine complex components into original image again
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            # crop to original size
            img_back = img_back[0:hh, 0:ww]

            # re-normalize to 8-bits in range of original
            min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
            notched = cv2.normalize(
                img_back,
                None,
                alpha=img_min,
                beta=img_max,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )

            # cv2.imshow("ORIGINAL", img)
            # # cv2.imshow("PADDED", imgp)
            # # cv2.imshow("MAG", mag)
            # # cv2.imshow("PHASE", phase)
            # cv2.imshow("SPECTRUM", spec)
            # cv2.imshow("THRESH", thresh)
            #
            # cv2.imshow("NOTCHED", notched)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            imgbr_res.append(notched)
            # return notched
            # # write result to disk
            # cv2.imwrite("pattern_lines_spectrum.png", (255*spec).clip(0,255).astype(np.uint8))
            # cv2.imwrite("pattern_lines_thresh.png", thresh)
            # cv2.imwrite("pattern_lines_mask.png", mask)
            # cv2.imwrite("pattern_lines_notched.png", notched)

        if not len(imgbr_res) == 0:
            imresult = cv2.merge([imgbr_res[0], imgbr_res[1], imgbr_res[2]])
            # cv2.imshow("MASK", mask)
            # cv2.imshow("Original", image)
            # cv2.imshow("Result", imresult)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            cv2.imwrite(imgPath, imresult)
        # cv2.imwrite(f'./filtered_images/filtered_{file}', imresult)
