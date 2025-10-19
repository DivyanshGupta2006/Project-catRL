from src.update_files import update_state
from src.position_sizing import portfolio_calculator
from src.utils import read_file, get_config

config = get_config.read_yaml()

def calculate_candle_metrics(candle):
    state = read_file.read_state()

    capital = config['strategy']['capital']
    current_portfolio_value = portfolio_calculator.calculate(candle)
    pnl = current_portfolio_value - capital

    state['metrics']['returns'] = pnl / capital
    state['metrics']['peak equity'] = max(current_portfolio_value, state['metrics']['peak equity'])
    state['metrics']['drawdown'] = (state['metrics']['peak equity'] - current_portfolio_value) / state['metrics']['peak equity']

    update_state.update(state)

def calculate_order_metrics(order):
    state = read_file.read_state()

    for crypto in order:
        state['metrics']['transaction_cost'] += config['strategy']['transaction_cost_fraction'] * order[crypto]['order_amt'] * order[crypto]['order_price']

    update_state.update(state)