from __future__ import annotations

import os
import random
import shutil
import sys
from enum import Enum
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from tools.config import Config

    Path = Union[str, bytes, os.PathLike]


class DataSource(Enum):
    TUEBINGEN = "/graphics/scratch/schuelej/sar/data"
    GRK = "/b1/data/endo_vids"


def get_video_filepath(data_source: DataSource, filename: Path) -> Path:
    return os.path.join(data_source.value, filename)


def make_folder(folder_name: Path) -> None:
    path = os.path.abspath(folder_name)

    if not os.path.isdir(path):
        try:
            print(f"\t ...{path}")
            os.makedirs(path)
        except Exception as e:
            print(e)
            sys.exit(1)


def make_folders(config: Config) -> None:
    print("Creating folders ...")
    for f in config.list_of_folders:
        make_folder(f)


def clone_data_folders(path: Path, dest: Path) -> None:
    shutil.copytree(path, dest)


def remove_data_folders(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def delete_files(list_of_files: List[Path]) -> None:
    for f in list_of_files:
        os.remove(f)


def get_random_video_path(base_path: Path) -> Path:
    generator = os.walk(base_path)
    path, subfolder_names = next(generator)[:2]

    if "cropped" not in subfolder_names:
        # choose random video from subfolder
        video_id = random.randint(1, len(subfolder_names)) - 1
        video_path = os.path.join(path, subfolder_names[video_id])
        path = get_random_video_path(video_path)

    return path


def get_random_raw_image(video_path: Path) -> Path:
    dir_raw = os.path.join(video_path, "raw")

    generator = os.walk(dir_raw)
    img_list = next(generator)[-1]
    img_id = random.randint(1, len(img_list)) - 1

    return img_list[img_id]
