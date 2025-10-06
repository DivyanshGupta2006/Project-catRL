import pandas as pd

from src.utils import get_config, get_absolute_path, check_dir, read_file

config = get_config.read_yaml()

def merge(type='training'):
    print('Merging...')
    symbols = config['data']['symbols']
    if type == 'training':
        data_dir = get_absolute_path.absolute(config['paths']['merged_training_data_directory'])
    elif type == 'val':
        data_dir = get_absolute_path.absolute(config['paths']['merged_val_data_directory'])
    elif type == 'test':
        data_dir = get_absolute_path.absolute(config['paths']['merged_test_data_directory'])
    check_dir.check(data_dir)
    merged_data = pd.DataFrame()
    for symbol in symbols:
        print(symbol)
        if type == 'training':
            data = read_file.read_preprocessed_training_data(symbol)
        elif type == 'val':
            data = read_file.read_preprocessed_val_data(symbol)
        elif type == 'test':
            data = read_file.read_preprocessed_test_data(symbol)
        symbol = symbol.split('/')[0]
        for col in data.columns:
            data.rename(columns={col : (col, symbol)}, inplace=True)
        if merged_data.empty:
            merged_data = data
        else:
            merged_data = pd.merge(merged_data, data, how="inner", left_index=True, right_index=True)

    merged_data.to_csv(data_dir / f"merged_{type}_data.csv")