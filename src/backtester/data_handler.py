import pandas as pd
import json

from src.update_files import update_state
from src.utils import get_config, read_file, get_absolute_path

config = get_config.read_yaml()

def fetch_data(data):
    state = read_file.read_state()
    current_timestep = pd.to_datetime(state['timestep'])
    state['timestep'] = current_timestep + pd.Timedelta(config['data']['timeframe'])
    state['timestep'] = state['timestep'].strftime("%Y-%m-%dT%H:%M:%SZ")
    update_state.update(state)
    return data.loc[current_timestep].copy()