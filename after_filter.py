from tools.config import Config, image_length
from tools.im2graph import extract_graphs
from tools.images import apply_img_mask, skeletonise_imgs, threshold_imgs
from video_data import frequency, is_synthetic, trim_times, video_filepath


def after_filter(conf: Config, skip_existing: bool) -> None:
    """
    Applies image processing functions to filtered images in the video directory.
    :param conf: video-dependent configuration
    :param skip_existing: False to overwrite existing graph file.
    """
    apply_img_mask(conf)
    threshold_imgs(conf)
    skeletonise_imgs(conf)
    extract_graphs(conf, skip_existing)


if __name__ == "__main__":
    print(f"Generating {image_length}px data for\n", f"\t{video_filepath}")
    conf = Config(
        video_filepath,
        frequency,
        img_length=image_length,
        trim_times=trim_times,
        synthetic=is_synthetic,
    )
    after_filter(conf, skip_existing=False)
