import os

from tools.config import Config, image_length
from tools.videos import make_video_clip
from video_data import frequency, is_synthetic, trim_times, video_filepath


def preview_folder_as_video(conf: Config, folder_name: str):
    if conf.has_trimmed:
        for section in conf.sections:
            source_path = section.__dict__[f"{folder_name}_img_folder"]

            section_path = os.path.dirname(source_path)
            section_name = os.path.basename(section_path)

            vid_filename = f"{folder_name}_{section_name}" + conf.ext
            target_fp = os.path.join(conf.base_folder, vid_filename)

            make_video_clip(source_path, target_fp, fps=conf.frequency)
    else:
        source_path = conf.__dict__[f"{folder_name}_img_folder"]
        vid_filename = folder_name + conf.ext
        target_fp = os.path.join(conf.base_folder, vid_filename)
        make_video_clip(source_path, target_fp, fps=conf.frequency)


if __name__ == "__main__":
    folder_names = ["cropped", "filtered", "overlay"]

    conf = Config(
        video_filepath,
        frequency,
        img_length=image_length,
        trim_times=trim_times,
        synthetic=is_synthetic,
        do_trim=False,
    )

    for folder in folder_names:
        print(f"Generating preview ({folder}) for\n", f"\t{video_filepath}")
        preview_folder_as_video(conf, folder)
