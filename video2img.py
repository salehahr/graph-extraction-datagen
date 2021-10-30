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
        # sys.exit()

        return success, image

    count = 0
    sec = 0
    frames_exist = True

    while frames_exist:
        sec = round(sec, 2)
        frames_exist, img = get_frame(sec)

        if frames_exist:
            filename = str(count).zfill(5)
            cv2.imwrite(os.path.join(img_folder, f'{filename}.png'), img)
        else:
            break

        count = count + 1
        sec = sec + (1 / frequency)

    print(f'{count} images were extracted into {img_folder}.')
