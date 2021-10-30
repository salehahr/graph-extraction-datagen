import os
import cv2


def video2img(vid_filename: str, img_folder: str, frequency: float = 25):
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
    minute_count = 0

    while frames_exist:
        seconds_total = round(seconds_total, 2)
        frames_exist, img = get_frame(seconds_total)

        if frames_exist:
            seconds_in_minute = seconds_total - (60 * minute_count)

            if seconds_in_minute >= 60:
                seconds_in_minute -= 60
                minute_count += 1

            filename = generate_frame_name(minute_count, seconds_in_minute)
            cv2.imwrite(os.path.join(img_folder, f'{filename}.png'), img)

        else:
            break

        count = count + 1
        seconds_total += (1 / frequency)

    print(f'{count} images were extracted into {img_folder}.')


def generate_frame_name(minute_count: int, seconds_in_minute: float) -> str:
    """
    Generates frame name in the format 0000_00000.png,
    where in the example '0001_00500.png', the frame was extracted at
    1 minute 500 milliseconds.
    """
    milliseconds = int(seconds_in_minute * 1000)
    return str(minute_count).zfill(4) + '_' + str(milliseconds).zfill(5)
