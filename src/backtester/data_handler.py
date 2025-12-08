from src.utils import get_config, convert

config = get_config.read_yaml()

def fetch_data(data, state):
    row = data.loc[data.index[state['timestep']]].copy()
    state['timestep'] += 1
    candle = convert.convert_to_dict(row)
    return candle, state