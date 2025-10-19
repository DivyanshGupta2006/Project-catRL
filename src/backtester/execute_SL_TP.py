from src.utils import get_config, read_file
from src.update_files import update_state, update_portfolio

config = get_config.read_yaml()

def execute(candle):
    portfolio = read_file.read_portfolio()
    state = read_file.read_state()

    for crypto in portfolio.index:
        # shorting
        if portfolio.loc[crypto, 'amt'] < 0:
            if candle[crypto]['high'] > portfolio.loc[crypto, 'stop_price']:
                portfolio.loc[crypto, 'amt'] -= portfolio.loc[crypto, 'stop_portion']
                state['cash'] += (portfolio.loc[crypto, 'stop_portion'] * portfolio.loc[crypto, 'stop_price'])

            elif candle[crypto]['low'] < portfolio.loc[crypto, 'take_price']:
                portfolio.loc[crypto, 'amt'] -= portfolio.loc[crypto, 'take_portion']
                state['cash'] += (portfolio.loc[crypto, 'take_portion'] * portfolio.loc[crypto, 'take_price'])

        # longing
        elif portfolio.loc[crypto, 'amt'] > 0:
            if candle[crypto]['low'] < portfolio.loc[crypto, 'stop_price']:
                portfolio.loc[crypto, 'amt'] -= portfolio.loc[crypto, 'stop_portion']
                state['cash'] += (portfolio.loc[crypto, 'stop_portion'] * portfolio.loc[crypto, 'stop_price'])

            elif candle[crypto]['high'] > portfolio.loc[crypto, 'take_price']:
                portfolio.loc[crypto, 'amt'] -= portfolio.loc[crypto, 'take_portion']
                state['cash'] += (portfolio.loc[crypto, 'take_portion'] * portfolio.loc[crypto, 'take_price'])

    update_portfolio.update(portfolio)
    update_state.update(state)