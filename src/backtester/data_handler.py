import pandas as pd
import json
from src.utils import get_config, read_file, get_absolute_path

config = get_config.read_yaml()

def fetch_data(data):
    state = read_file.read_state()
    dir = get_absolute_path.absolute(config['paths']['state_directory'])
    timestep = state['timestep']
    state['timestep'] = pd.to_datetime(state['timestep']) + pd.Timedelta(config['data']['timeframe'])
    with open(dir, 'w') as f:
        json.dump(state, f, indent=4)
    return data.loc[timestep]
