import pandas as pd

from src.utils import get_config, get_absolute_path, check_dir, read_file

config = get_config.read_yaml()

def merge_normalized(type='training'):
    print(f'Merging {type} normalized data...')
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
        if type == 'training':
            data = read_file.read_preprocessed_training_data(symbol)
        elif type == 'val':
            data = read_file.read_preprocessed_val_data(symbol)
        elif type == 'test':
            data = read_file.read_preprocessed_test_data(symbol)
        symbol = symbol.split('/')[0]
        for col in data.columns:
            data.rename(columns={col : (col,symbol)}, inplace=True)
        if merged_data.empty:
            merged_data = data
        else:
            merged_data = pd.merge(merged_data, data, how="inner", left_index=True, right_index=True)

    merged_data.to_csv(data_dir / f"merged_{type}_normalized_data.csv")

def merge_unnormalized(type='training'):
    print(f'Merging {type} unnormalized data...')

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
        if type == 'training':
            data = read_file.read_featured_training_data(symbol)
        elif type == 'val':
            data = read_file.read_featured_val_data(symbol)
        elif type == 'test':
            data = read_file.read_featured_test_data(symbol)
        symbol = symbol.split('/')[0]
        for col in data.columns:
            data.rename(columns={col: (col, symbol)}, inplace=True)
        if merged_data.empty:
            merged_data = data
        else:
            merged_data = pd.merge(merged_data, data, how="inner", left_index=True, right_index=True)

    merged_data.dropna(inplace=True)

    if type == 'training':
        merged_data = merged_data.loc['2020-10-01 00:00:00':'2024-09-29 23:00:00']
    elif type == 'val':
        merged_data = merged_data.loc['2024-10-01 00:00:00':'2025-03-30 23:00:00']
    else:
        merged_data = merged_data.loc['2025-04-01 00:00:00':'2025-09-30 23:00:00']

    merged_data.to_csv(data_dir / f"merged_{type}_unnormalized_data.csv")