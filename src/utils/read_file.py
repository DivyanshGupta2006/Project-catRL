import pandas as pd
from src.utils import get_absolute_path, get_config

config = get_config.read_yaml()

def read_raw_data(file) -> pd.DataFrame:
    dir = get_absolute_path.absolute(config['paths']['raw_data_directory'])
    print(f"Reading Raw Equity data: {file} !")
    try:
        data = pd.read_csv(f'{dir}/{file}.csv', index_col=[0])
        print(f"Successfully read: {file} !")
        return data
    except Exception as e:
        print(f"Failed to read {file}: {e} !")
        return None