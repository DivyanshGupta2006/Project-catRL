from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def get_project_root():
    return PROJECT_ROOT

def absolute(path):
    return get_project_root().joinpath(path)