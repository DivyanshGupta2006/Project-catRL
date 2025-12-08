import pandas as pd
import json
import os

from src.utils import get_absolute_path, get_config

config = get_config.read_yaml()
n = config['hyperparameters']['num_assets']
paths = config['paths']

def _get_path(key, file):
    directory = get_absolute_path.absolute(paths[key])
    return os.path.join(directory, file)

def _read_csv(file):
    try:
        return pd.read_csv(file, index_col=[0], engine='pyarrow')
    except Exception as e:
        print(e)
        return None

def read_raw_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('raw_data_directory', file_name)
    return _read_csv(path)

def read_raw_training_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('raw_training_data_directory', file_name)
    return _read_csv(path)

def read_raw_test_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('raw_test_data_directory', file_name)
    return _read_csv(path)

def read_raw_val_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('raw_val_data_directory', file_name)
    return _read_csv(path)

def read_featured_training_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('featured_training_data_directory', file_name)
    return _read_csv(path)

def read_featured_test_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('featured_test_data_directory', file_name)
    return _read_csv(path)

def read_featured_val_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('featured_val_data_directory', file_name)
    return _read_csv(path)

def read_preprocessed_training_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('processed_training_data_directory', file_name)
    return _read_csv(path)

def read_preprocessed_test_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('processed_test_data_directory', file_name)
    return _read_csv(path)

def read_preprocessed_val_data(symbol):
    file_name = f"{symbol.split('/')[0]}.csv"
    path = _get_path('processed_val_data_directory', file_name)
    return _read_csv(path)

def read_merged_training_data(normalized = True):
    file_name = 'merged_training_normalized_data.csv' if normalized else 'merged_training_unnormalized_data.csv'
    path = _get_path('merged_training_data_directory', file_name)
    return _read_csv(path)

def read_merged_test_data(normalized = True):
    file_name = 'merged_test_normalized_data.csv' if normalized else 'merged_test_unnormalized_data.csv'
    path = _get_path('merged_test_data_directory', file_name)
    return _read_csv(path)

def read_merged_val_data(normalized = True):
    file_name = 'merged_val_normalized_data.csv' if normalized else 'merged_val_unnormalized_data.csv'
    path = _get_path('merged_val_data_directory', file_name)
    return _read_csv(path)

def read_portfolio():
    dir = get_absolute_path.absolute(paths['portfolio_directory'])
    try:
        return pd.read_csv(dir, index_col=['symbol'])
    except Exception as e:
        return pd.DataFrame()

def read_state():
    dir = get_absolute_path.absolute(config['paths']['state_directory'])
    try:
        with open(dir, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}