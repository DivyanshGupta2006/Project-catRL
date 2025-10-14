import os
import pandas as pd
from src.utils import get_config, get_absolute_path, read_file

config = get_config.read_yaml()

filename = get_absolute_path.absolute(config['paths']['portfolio_directory'])
symbols = config['data']['symbols']

def check_portfolio_csv():
    """
    Creates a CSV file with a specified index and columns if it does not already exist.
    """
    index_col = []
    for symbol in symbols:
        index_col.append(symbol.split('/')[0])

    other_cols = ['holding','order_price(USDT)','stop_price(USDT)','stop_quantity','take_price(USDT)','take_quantity']

    # check if the file already exists in the current directory
    if os.path.exists(filename):
        return

    # create if it does not exist
    try:
        print(f"File '{filename}' not found. Creating a new file...")
        all_columns = ['symbol'] + other_cols
        df = pd.DataFrame(columns=all_columns)
        df['symbol'] = index_col
        df.set_index('symbol', inplace=True)
        df.fillna(0, inplace=True)
        df.to_csv(filename)
        print(f"Successfully created '{filename}' with '{index_col}' as the index.")
    except Exception as e:
        print(e)

def calc_portfolio():
    check_portfolio_csv()
    df = pd.read_csv(filename)
    current_values = df['holding'] * df['order_price(USDT)']
    cur_holding = current_values.sum()
    state = read_file.read_state()
    cur_holding += state["cash"]
    return cur_holding