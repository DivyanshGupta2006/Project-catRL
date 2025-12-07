import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def get_project_root():
    return PROJECT_ROOT

def absolute(path):
    return get_project_root().joinpath(path)

def join_path(directory, file, type):
    file = f'{file}.{type}'
    return os.path.join(directory, file)