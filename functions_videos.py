import os

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip


def generate_time_tag(time_in_s: float) -> str:
    """
    Generates time tag in the format 0000_00000,
    where the example '0001_00500', denotes
    1 minute 500 milliseconds.
    """
    minute, seconds = divmod(time_in_s, 60)
    millisecs = int(seconds * 1000)
    return str(int(minute)).zfill(4) + '_' + str(millisecs).zfill(5)


def video2img(vid_filename: str, img_folder: str, frequency: float = 25) -> None:
    """
    Saves video frames as .png images.

    :param frequency: number of frames per second
    """

    cap = cv2.VideoCapture(vid_filename)

    def get_frame(seconds: float):
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        success, image = cap.read()

        # cv2.imshow('title', image_cropped)
        # cv2.waitKey()

        return success, image

    count = 0
    frames_exist = True

    seconds_total = 0

    while frames_exist:
        seconds_total = round(seconds_total, 2)
        frames_exist, img = get_frame(seconds_total)

        if frames_exist:
            filename = generate_time_tag(seconds_total)
            cv2.imwrite(os.path.join(img_folder, f'{filename}.png'), img)

        else:
            break

        count = count + 1
        seconds_total += (1 / frequency)

    print(f'{count} images were extracted into {img_folder}.')
