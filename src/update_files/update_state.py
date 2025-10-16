import json

from src.utils import get_config, get_absolute_path

config = get_config.read_yaml()

def update(state):
    dir = get_absolute_path.absolute(config['paths']['state_directory'])
    with open(dir, 'w') as f:
        json.dump(state, f, indent=4)