import pandas as pd

from src.utils import get_absolute_path,get_config

def rebalance(order):
    config = get_config.read_yaml()
    file = 'portfolio.csv'
    dir  = get_absolute_path.absolute(config['paths']['portfolio_directory'])
    data = pd.DataFrame.from_dict(order,orient='index')
    try:
        data.to_csv(dir)
        print(f"\nSuccessfully saved orders to '{file}'")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")

