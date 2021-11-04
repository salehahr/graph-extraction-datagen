import os
import shutil
import sys


def make_folder(folder_name: str) -> None:
    path = os.path.abspath(folder_name)

    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)
            sys.exit(1)


def make_folders(config):
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
