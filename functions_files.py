import os


def make_folder(folder_name: str) -> None:
    path = os.path.join(os.getcwd(), folder_name)
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
