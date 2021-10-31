import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions_files import make_folders, remove_data_folders, delete_files
from functions_videos import video2img