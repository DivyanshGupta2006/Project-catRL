from src.update_files import update_state
from src.utils import get_config, read_file, convert

config = get_config.read_yaml()

def fetch_data(data):
    state = read_file.read_state()
    row = data.loc[data.index[state['timestep']]].copy()
    state['timestep'] += 1
    update_state.update(state)
    candle = convert.convert_to_dict(row)
    return candle