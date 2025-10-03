from src.utils import get_config, get_absolute_path, check_dir, read_file

config = get_config.read_yaml()

def preprocess():
    print("Preprocessing data....")
    symbols = config['data']['symbols']
    for symbol in symbols:
        pass