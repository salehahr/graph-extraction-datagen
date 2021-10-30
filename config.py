import os
from functions_files import make_folder

VIDEO_FILENAME_EXT = 'data/GRK021_test.mp4'
VIDEO_FILENAME = os.path.splitext(VIDEO_FILENAME_EXT)[0]

raw_img_folder = f'{VIDEO_FILENAME}/raw'
cropped_img_folder = f'{VIDEO_FILENAME}/cropped'
filtered_img_folder = f'{VIDEO_FILENAME}/filtered'
masked_img_folder = f'{VIDEO_FILENAME}/masked'
threshed_img_folder = f'{VIDEO_FILENAME}/threshed'
preproc_img_folder = f'{VIDEO_FILENAME}/skeleton'
landmarks_img_folder = f'{VIDEO_FILENAME}/landmarks'
poly_graph_img_folder = f'{VIDEO_FILENAME}/poly_graph'
overlay_img_folder = f'{VIDEO_FILENAME}/overlay'


def make_folders():
    make_folder(raw_img_folder)
    make_folder(cropped_img_folder)
    make_folder(filtered_img_folder)
    make_folder(masked_img_folder)
    make_folder(threshed_img_folder)
    make_folder(preproc_img_folder)
    make_folder(landmarks_img_folder)
    make_folder(poly_graph_img_folder)
    make_folder(overlay_img_folder)


# vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

crop_top, crop_bottom = 4, 1080
crop_left, crop_right = 416, 1532

crop_height = crop_bottom - crop_top
crop_width = crop_right - crop_left
