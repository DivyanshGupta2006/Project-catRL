from src.update_files import update_state, update_portfolio
from src.utils import read_file, get_config

config = get_config.read_yaml()

def execute(order):
    portfolio = read_file.read_portfolio()
    state = read_file.read_state()
    for crypto in order:
        portfolio.loc[crypto, 'amt'] += order[crypto]['order_amt']
        portfolio.loc[crypto, 'order_price'] = order[crypto]['order_price']
        portfolio.loc[crypto, 'stop_price'] = order[crypto]['stop_price']
        portfolio.loc[crypto, 'take_price'] = order[crypto]['take_price']
        portfolio.loc[crypto, 'stop_portion'] = order[crypto]['stop_portion']
        portfolio.loc[crypto, 'take_portion'] = order[crypto]['take_portion']
        state['cash'] -= (order[crypto]['order_price'] * order[crypto]['order_amt'] * (1 + config['strategy']['transaction_cost_fraction']))

    update_portfolio.update(portfolio)
    update_state.update(state)