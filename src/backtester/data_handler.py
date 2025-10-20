import ast

from src.update_files import update_state
from src.utils import get_config, read_file

config = get_config.read_yaml()

def fetch_data(data):
    state = read_file.read_state()
    row = data.loc[data.index[state['timestep']]].copy()
    state['timestep'] += 1
    update_state.update(state)
    row.index = row.index.map(ast.literal_eval)
    candle_df = row.unstack(level=0)
    candle = candle_df.to_dict(orient='index')
    return candle