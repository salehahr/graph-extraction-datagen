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


def video2img(config, frequency: float = 25) -> None:
    """
    Saves video frames as .png images.

    :param frequency: number of frames per second
    """

    cap = cv2.VideoCapture(config.filepath)

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
            cv2.imwrite(os.path.join(config.raw_img_folder, f'{filename}.png'), img)

        else:
            break

        count = count + 1
        seconds_total += (1 / frequency)

    print(f'{count} images were extracted into {config.raw_img_folder}.')


def trim_video(orig: str, start: float, end: float, target: str = None) -> None:
    """
    Trims video.
    :param orig: original video filepath
    :param start: start trim in s
    :param end: end trim in s
    :param target: target video filepath
    :return:
    """
    start_tag = generate_time_tag(start)
    end_tag = generate_time_tag(end)

    temp_name, ext = os.path.splitext(orig)
    target = temp_name + '_' + start_tag + '__' + end_tag + ext \
        if target is None else target

    video = VideoFileClip(orig).subclip(start, end)
    video.write_videofile(target)

    # # produces green artifacts
    # ffmpeg_extract_subclip(orig_filename,
    #                        start_time_in_s, end_time_in_s,
    #                        targetname=target_filename)
