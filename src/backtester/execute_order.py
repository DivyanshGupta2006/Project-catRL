from src.utils import get_config

config = get_config.read_yaml()

def execute(order, state, portfolio):
    for crypto in order:
        portfolio.loc[crypto, 'amt'] += order[crypto]['order_amt']
        portfolio.loc[crypto, 'order_price'] = order[crypto]['order_price']
        portfolio.loc[crypto, 'stop_price'] = order[crypto]['stop_price']
        portfolio.loc[crypto, 'take_price'] = order[crypto]['take_price']
        portfolio.loc[crypto, 'stop_portion'] = order[crypto]['stop_portion']
        portfolio.loc[crypto, 'take_portion'] = order[crypto]['take_portion']
        state['cash'] -= (order[crypto]['order_price'] * order[crypto]['order_amt'] * (1 + config['strategy']['transaction_cost_fraction']))

    return state, portfolio