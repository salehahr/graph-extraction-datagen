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


def generate_time_tag_from_interval(interval: list) -> str:
    start_tag = generate_time_tag(interval[0])
    end_tag = generate_time_tag(interval[1])
    return start_tag + '__' + end_tag


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


def trim_video_section(orig: str, interval: list, target: str = None) -> str:
    """
    Extracts a section from a video.
    :param orig: original video filepath
    :param interval: list of start trim in s and end trim in s
    :param target: target video filepath
    :return: trimmed video filepath
    """
    time_tag = generate_time_tag_from_interval(interval)

    base_name, ext = os.path.splitext(orig)
    target = base_name + '_' + time_tag + ext \
        if target is None else target

    if not os.path.isfile(target):
        video = VideoFileClip(orig).subclip(interval[0], interval[1])
        video.write_videofile(target)

    # # produces green artifacts
    # ffmpeg_extract_subclip(orig_filename,
    #                        start_time_in_s, end_time_in_s,
    #                        targetname=target_filename)

    return target


def trim_video(config, targets: list = []):
    """
    Extracts video sections according to the intervals specified in config.
    :param config:
    :param targets: list of target filepaths
    :return: list of filepaths of the trimmed videos
    """
    if config.trim_times is None:
        return

    if targets:
        assert (len(config.trim_times) == len(targets))
        return [trim_video_section(config.filepath,
                                   interval,
                                   target=targets[i]) for i, interval in enumerate(config.trim_times)]
    else:
        return [trim_video_section(config.filepath,
                                   interval) for interval in config.trim_times]

