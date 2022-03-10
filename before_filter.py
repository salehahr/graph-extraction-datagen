# * Extracts raw images from video
# * Crops the video stills

from preview import preview_folder_as_video
from tools.config import Config, image_length
from tools.files import make_folders
from tools.images import crop_imgs, fft_filter_vert_stripes
from tools.videos import video2img
from video_data import frequency, is_synthetic, trim_times, video_filepath, use_images, fft_filter


def before_filter(conf=None):
    if not conf.use_images:
        for section in conf.sections:
            make_folders(section)
            video2img(section)


    crop_imgs(conf)
    if fft_filter:
        fft_filter_vert_stripes(conf)

    assert len(conf.raw_image_files) == len(conf.cropped_image_files)


if __name__ == "__main__":
    print(f"Generating {image_length}px data for\n", f"\t{video_filepath}")
    conf = Config(
        video_filepath,
        frequency,
        use_images=use_images,
        img_length=image_length,
        trim_times=trim_times,
        synthetic=is_synthetic,

    )
    before_filter(conf)
    if not conf.use_images:
        preview_folder_as_video(conf, "cropped")

