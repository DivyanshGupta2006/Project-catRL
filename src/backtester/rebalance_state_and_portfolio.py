from src.update_files import update_portfolio, update_state
from src.utils import read_file

def rebalance(candle, order):
    portfolio = read_file.read_portfolio()
    state = read_file.read_state()

    for crypto in portfolio.index:

        if crypto in order.keys():
            portfolio.loc[crypto,'stop_price'] = order[crypto]['stop_price']
            portfolio.loc[crypto,'take_price'] = order[crypto]['take_price']
            portfolio.loc[crypto,'stop_portion'] = order[crypto]['stop_portion']
            portfolio.loc[crypto,'take_portion'] = order[crypto]['take_portion']

            portfolio.loc[crypto,'amt'] += order[crypto]['order_amt']

            state['cash'] -= order[crypto]['order_amt'] * order[crypto]['order_price']

    update_state.update(state)
    update_portfolio.update(portfolio)