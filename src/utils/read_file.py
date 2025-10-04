import pandas as pd
from src.utils import get_absolute_path, get_config

config = get_config.read_yaml()

def read_raw_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['raw_data_directory'])
    file = symbol.split('/')[0]
    print(f"Reading Raw data: {file} !")
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        print(f"Successfully read: {file} !")
        return data
    except Exception as e:
        print(f"Failed to read {file}: {e} !")
        return None

def read_raw_training_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['raw_training_data_directory'])
    file = symbol.split('/')[0]
    print(f"Reading Training data: {file} !")
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        print(f"Successfully read: {file} !")
        return data
    except Exception as e:
        print(f"Failed to read {file}: {e} !")
        return None

def read_raw_test_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['raw_test_data_directory'])
    file = symbol.split('/')[0]
    print(f"Reading Test data: {file} !")
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        print(f"Successfully read: {file} !")
        return data
    except Exception as e:
        print(f"Failed to read {file}: {e} !")
        return None

def read_raw_val_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['raw_val_data_directory'])
    file = symbol.split('/')[0]
    print(f"Reading Validation data: {file} !")
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        print(f"Successfully read: {file} !")
        return data
    except Exception as e:
        print(f"Failed to read {file}: {e} !")
        return None