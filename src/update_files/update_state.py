import json

from src.utils import get_config, get_absolute_path, read_file

config = get_config.read_yaml()

def update(state):
    dir = get_absolute_path.absolute(config['paths']['state_directory'])
    with open(dir, 'w') as f:
        json.dump(state, f, indent=4)

def set_state(data, capital):
    start = data.index[0]
    state = read_file.read_state()
    state['timestep'] = start
    state['cash'] = capital
    state['transaction_cost'] = 0.0
    state['returns'] = 0.0
    state['drawdown'] = 0.0
    state['sharpe ratio'] = 0.0
    update(state)