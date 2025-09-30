import yaml
from src.utils import get_absolute_path

def read_yaml() -> dict:
    config_path = get_absolute_path.get_project_root().joinpath('config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config