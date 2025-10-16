import pandas as pd

from src.utils import get_absolute_path,get_config

config = get_config.read_yaml()

def rebalance_portfolio(order):
    dir  = get_absolute_path.absolute(config['paths']['portfolio_directory'])
    data = pd.DataFrame.from_dict(order,orient='index')
    try:
        data.to_csv(dir)
    except Exception as e:
        pass

def rebalance_state(order):
    dir = get_absolute_path.absolute(config['paths']['portfolio_directory'])