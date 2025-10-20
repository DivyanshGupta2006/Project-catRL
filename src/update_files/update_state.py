import json

from src.utils import get_config, get_absolute_path, read_file

config = get_config.read_yaml()

def update(state):
    dir = get_absolute_path.absolute(config['paths']['state_directory'])
    with open(dir, 'w') as f:
        json.dump(state, f, indent=4)

def set_state(capital):
    state = read_file.read_state()
    state['timestep'] = 0
    state['cash'] = capital
    state['metrics'] = {'transaction_cost': 0.0,
        'returns': 0.0,
        'peak equity': 0.0,
        'drawdown': 0.0}
    update(state)