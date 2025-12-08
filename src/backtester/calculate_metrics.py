from src.position_sizing import portfolio_calculator
from src.utils import get_config

config = get_config.read_yaml()

def calculate_candle_metrics(candle, state, portfolio):
    capital = config['strategy']['capital']
    current_portfolio_value = portfolio_calculator.calculate(candle, state, portfolio)
    pnl = current_portfolio_value - capital

    state['metrics']['returns'] = pnl / capital
    state['metrics']['peak equity'] = max(current_portfolio_value, state['metrics']['peak equity'])
    state['metrics']['drawdown'] = (state['metrics']['peak equity'] - current_portfolio_value) / state['metrics']['peak equity']

    return state

def calculate_order_metrics(order, state):

    for crypto in order:
        state['metrics']['transaction_cost'] += abs(config['strategy']['transaction_cost_fraction'] * order[crypto]['order_amt'] * order[crypto]['order_price'])

    return state