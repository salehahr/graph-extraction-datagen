import os
import random
import shutil
import sys


def make_folder(folder_name: str) -> None:
    path = os.path.abspath(folder_name)

    if not os.path.isdir(path):
        try:
            print(f"\t ...{path}")
            os.makedirs(path)
        except Exception as e:
            print(e)
            sys.exit(1)


def make_folders(config):
    print("Creating folders ...")
    for f in config.list_of_folders:
        make_folder(f)


def clone_data_folders(path, dest):
    shutil.copytree(path, dest)


def remove_data_folders(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def delete_files(list_of_files):
    for f in list_of_files:
        os.remove(f)


def get_random_video_path(base_path):
    generator = os.walk(base_path)
    path, subfolder_names = next(generator)[:2]

    if "raw" not in subfolder_names:
        # choose random video from subfolder
        video_id = random.randint(1, len(subfolder_names)) - 1
        video_path = os.path.join(path, subfolder_names[video_id])
        path = get_random_video_path(video_path)

    return path


def get_random_raw_image(video_path):
    dir_raw = os.path.join(video_path, "raw")

    generator = os.walk(dir_raw)
    img_list = next(generator)[-1]
    img_id = random.randint(1, len(img_list)) - 1

    return img_list[img_id]
