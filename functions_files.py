import os
import shutil


def make_folder(folder_name: str) -> None:
    path = os.path.abspath(folder_name)
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def make_folders(list_of_folders: list):
    for f in list_of_folders:
        make_folder(f)


def remove_data_folders(path):
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass


def delete_files(list_of_files):
    for f in list_of_files:
        os.remove(f)
