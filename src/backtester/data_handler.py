import pandas as pd
import json

from src.update_files import update_state
from src.utils import get_config, read_file, get_absolute_path

config = get_config.read_yaml()

def fetch_data(data):
    state = read_file.read_state()
    timestep = state['timestep']
    state['timestep'] = pd.to_datetime(state['timestep']) + pd.Timedelta(config['data']['timeframe'])
    update_state.update(state)
    return data.loc[timestep]