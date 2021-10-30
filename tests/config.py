import os

VIDEO_FULL_FILEPATH_EXT = 'short_video.mp4'
DATA_FILEPATH = os.path.splitext(VIDEO_FULL_FILEPATH_EXT)[0]
assert(os.path.isfile(VIDEO_FULL_FILEPATH_EXT))

raw_img_folder = f'{DATA_FILEPATH}/raw'
cropped_img_folder = f'{DATA_FILEPATH}/cropped'
filtered_img_folder = f'{DATA_FILEPATH}/filtered'
masked_img_folder = f'{DATA_FILEPATH}/masked'
threshed_img_folder = f'{DATA_FILEPATH}/threshed'
preproc_img_folder = f'{DATA_FILEPATH}/skeleton'
landmarks_img_folder = f'{DATA_FILEPATH}/landmarks'
poly_graph_img_folder = f'{DATA_FILEPATH}/poly_graph'
overlay_img_folder = f'{DATA_FILEPATH}/overlay'

list_of_folders = [
    raw_img_folder,
    cropped_img_folder,
    filtered_img_folder,
    masked_img_folder,
    threshed_img_folder,
    preproc_img_folder,
    landmarks_img_folder,
    poly_graph_img_folder,
    overlay_img_folder,
]

# vid_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

crop_top, crop_bottom = 4, 1080
crop_left, crop_right = 416, 1532

crop_height = crop_bottom - crop_top
crop_width = crop_right - crop_left
