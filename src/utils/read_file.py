import joblib
import pandas as pd
import json
from src.utils import get_absolute_path, get_config

config = get_config.read_yaml()

def read_raw_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['raw_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_raw_training_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['raw_training_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_raw_test_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['raw_test_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_raw_val_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['raw_val_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_featured_training_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['featured_training_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_featured_test_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['featured_test_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_featured_val_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['featured_val_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_preprocessor(symbol):
    dir = get_absolute_path.absolute(config['paths']['preprocessor_directory'])
    file = symbol.split('/')[0]
    try:
        pre = joblib.load(f'{dir}/{file}.joblib')
        return pre
    except Exception as e:
        return None

def read_preprocessed_training_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['processed_training_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_preprocessed_test_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['processed_test_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_preprocessed_val_data(symbol) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['processed_val_data_directory'])
    file = symbol.split('/')[0]
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_merged_training_data() -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['merged_training_data_directory'])
    file = 'merged_training_data'
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_merged_test_data() -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['merged_test_data_directory'])
    file = 'merged_test_data'
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_merged_val_data() -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['merged_val_data_directory'])
    file = 'merged_val_data'
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        return data
    except Exception as e:
        return None

def read_portfolio():
    dir = get_absolute_path.absolute(config['paths']['portfolio_directory'])
    try:
        data = pd.read_csv(dir, index_col=[0])
        return data
    except Exception as e:
        return None

def read_state():
    dir = get_absolute_path.absolute(config['paths']['state_directory'])
    try:
        with open(dir, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return None

