import pandas as pd
import numpy as np
from src.utils import get_config, get_absolute_path, check_dir, read_file

config = get_config.read_yaml()

def preprocess(type='training', to_normalize=True):
    print(f"Preprocessing data for {type}....")
    symbols = config['data']['symbols']
    if type == 'training':
        data_dir = get_absolute_path.absolute(config['paths']['processed_training_data_directory'])
    elif type == 'val':
        data_dir = get_absolute_path.absolute(config['paths']['processed_val_data_directory'])
    elif type == 'test':
        data_dir = get_absolute_path.absolute(config['paths']['processed_test_data_directory'])

    for symbol in symbols:
        if type == 'training':
            data = read_file.read_featured_training_data(symbol)
            data_dir = get_absolute_path.absolute(config['paths']['processed_training_data_directory'])
        elif type == 'val':
            data = read_file.read_featured_val_data(symbol)
            data_dir = get_absolute_path.absolute(config['paths']['processed_val_data_directory'])
        elif type == 'test':
            data = read_file.read_featured_test_data(symbol)
            data_dir = get_absolute_path.absolute(config['paths']['processed_test_data_directory'])

        scaled_data = data.copy()

        if to_normalize:
            for col in data.columns:
                rolling_min = data[col].rolling(window=config['data']['normalization_window'],
                                           min_periods=config['data']['normalization_window']).min()
                rolling_max = data[col].rolling(window=config['data']['normalization_window'],
                                           min_periods=config['data']['normalization_window']).max()
                rolling_range = rolling_max - rolling_min + 1e-9

                scaled_data[col] = (data[col] - rolling_min) / rolling_range

        if type == 'training':
            data = scaled_data[data.index >= config['data']['begin_train_date']].copy()
        elif type == 'val':
            data = scaled_data[data.index >= config['data']['begin_val_date']].copy()
        elif type == 'test':
            data = scaled_data[data.index >= config['data']['begin_test_date']].copy()

        # Handling NaN values
        data = data.ffill()

        symbol = symbol.split('/')[0]
        data.to_csv(get_absolute_path.join_path(data_dir, symbol, 'csv'))
