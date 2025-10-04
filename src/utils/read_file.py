import joblib
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
    print(f"Reading Raw Training data: {file} !")
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
    print(f"Reading Raw Test data: {file} !")
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
    print(f"Reading Raw Validation data: {file} !")
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        print(f"Successfully read: {file} !")
        return data
    except Exception as e:
        print(f"Failed to read {file}: {e} !")
        return None

def read_featured_training_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['featured_training_data_directory'])
    file = symbol.split('/')[0]
    print(f"Reading Featured Training data: {file} !")
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        print(f"Successfully read: {file} !")
        return data
    except Exception as e:
        print(f"Failed to read {file}: {e} !")
        return None

def read_featured_test_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['featured_test_data_directory'])
    file = symbol.split('/')[0]
    print(f"Reading Featured Test data: {file} !")
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        print(f"Successfully read: {file} !")
        return data
    except Exception as e:
        print(f"Failed to read {file}: {e} !")
        return None

def read_featured_val_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['featured_val_data_directory'])
    file = symbol.split('/')[0]
    print(f"Reading Featured Validation data: {file} !")
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        print(f"Successfully read: {file} !")
        return data
    except Exception as e:
        print(f"Failed to read {file}: {e} !")
        return None

def read_preprocessor(symbol):
    dir = get_absolute_path.absolute(config['paths']['preprocessor_directory'])
    file = symbol.split('/')[0]
    print(f"Reading Preprocessor data: {file} !")
    try:
        pre = joblib.load(f'{dir}/{file}.joblib')
        print(f"Successfully loaded: {file} !")
        return pre
    except Exception as e:
        print(f"Failed to load {file}: {e} !")
        return None