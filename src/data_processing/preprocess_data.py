from src.utils import get_config, get_absolute_path, check_dir, read_file

config = get_config.read_yaml()

def preprocess(type='training'):
    print("Preprocessing data....")
    symbols = config['data']['symbols']
    for symbol in symbols:
        if type == 'training':
            data = read_file.read_featured_training_data(symbol)
        elif type == 'val':
            data = read_file.read_featured_val_data(symbol)
        elif type == 'test':
            data = read_file.read_featured_test_data(symbol)