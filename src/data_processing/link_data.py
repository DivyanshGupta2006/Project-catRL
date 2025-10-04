import pandas as pd
from src.utils import get_config, get_absolute_path, read_file

config = get_config.read_yaml()

def link(data = 'val-test'):
    symbols = config['data']['symbols']
    for symbol in symbols:
        if data == 'training-val':
            data1 = read_file.read_raw_training_data(symbol)
            data2 = read_file.read_raw_val_data(symbol)
            data1 = data1.tail(config['data']['backdate'])
            data2 = pd.concat([data1, data2])
            data_dir = get_absolute_path.absolute(config['paths']['raw_val_data_directory'])
            symbol = symbol.split('/')[0]
            path = f'{symbol}.csv'
            data2.to_csv(data_dir / path)
        else:
            data1 = read_file.read_raw_val_data(symbol)
            data2 = read_file.read_raw_test_data(symbol)
            data1 = data1.tail(config['data']['backdate'])
            data2 = pd.concat([data1, data2])
            data_dir = get_absolute_path.absolute(config['paths']['raw_test_data_directory'])
            symbol = symbol.split('/')[0]
            path = f'{symbol}.csv'
            data2.to_csv(data_dir / path)