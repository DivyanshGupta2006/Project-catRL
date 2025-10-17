import pandas as pd

from src.utils import get_config, read_file
from src.update_files import update_state, update_portfolio

config = get_config.read_yaml()

def execute(candle):
    portfolio = read_file.read_portfolio()
    state = read_file.read_state()

    for crypto in portfolio.index:
        # shorting
        if portfolio.loc[crypto, 'stop_price'] < 0:
            if candle[('high', crypto)] > portfolio.loc[crypto, 'stop_price']:
                state['cash'] -= portfolio.loc[crypto, 'stop_price'] * portfolio.loc[crypto, 'stop_portion']
                portfolio.loc[crypto, 'stop_price'] += portfolio.loc[crypto, 'stop_portion']

            elif candle[('low', crypto)] < portfolio.loc[crypto, 'take_price']:
                state['cash'] -= portfolio.loc[crypto, 'take_price'] * portfolio.loc[crypto, 'take_portion']
                portfolio.loc[crypto, 'stop_price'] += portfolio.loc[crypto, 'take_portion']

        # longing
        elif portfolio.loc[crypto, 'stop_price'] > 0:
            if candle[('low', crypto)] < portfolio.loc[crypto, 'stop_price']:
                state['cash'] += portfolio.loc[crypto, 'stop_price'] * portfolio.loc[crypto, 'stop_portion']
                portfolio.loc[crypto, 'stop_price'] -= portfolio.loc[crypto, 'stop_portion']

            elif candle[('high', crypto)] > portfolio.loc[crypto, 'take_price']:
                state['cash'] += portfolio.loc[crypto, 'take_price'] * portfolio.loc[crypto, 'take_portion']
                portfolio.loc[crypto, 'stop_price'] -= portfolio.loc[crypto, 'take_portion']

    # this should be done before placing orders
    update_portfolio.update(portfolio)
    update_state.update(state)