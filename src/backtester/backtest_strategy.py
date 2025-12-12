from tqdm import tqdm
import matplotlib.pyplot as plt

from src.backtester import data_handler, place_order, execute_order, execute_SL_TP, print_backtesting_results, calculate_metrics
from src.strategy import predict_position
from src.risk_management import slippage, stop_loss, take_profit
from src.position_sizing import amount_calculator, portfolio_calculator
from src.update_files import update_state, update_portfolio
from src.utils import read_file, get_config, get_absolute_path

config = get_config.read_yaml()

def backtest_step(data, equity_curve, state, portfolio):
    candle, state = data_handler.fetch_data(data, state)
    _, state, portfolio = execute_SL_TP.execute(candle, state, portfolio)
    state = calculate_metrics.calculate_candle_metrics(candle, state, portfolio)
    candle = predict_position.predict(candle)
    Pt = portfolio_calculator.calculate(candle, state, portfolio)
    if Pt < (0.001 * config['strategy']['capital']):
        print("Bankruptcy!")
        return 0
    equity_curve.append(Pt)
    candle = slippage.get_order_price(candle, Pt, portfolio)
    candle = amount_calculator.calculate(candle, Pt, portfolio)
    candle = stop_loss.get_stop_loss(candle)
    candle = take_profit.get_take_profit(candle)
    order = place_order.place(candle, portfolio)
    state, portfolio = execute_order.execute(order, state, portfolio)
    state = calculate_metrics.calculate_order_metrics(order, state)
    return 1, state, portfolio

def backtest(data, label):
    update_state.set_state(config['strategy']['capital'])
    update_portfolio.set_portfolio()
    state = read_file.read_state()
    portfolio = read_file.read_portfolio()
    equity_curve = []
    res = 1
    for _ in tqdm(range(len(data) - 1), desc="Backtesting Progress"):
        res, state, portfolio = backtest_step(data, equity_curve, state, portfolio)
        if res == 0:
            break

    # last row
    if res == 1:
        candle, state = data_handler.fetch_data(data, state)
        _, state, portfolio = execute_SL_TP.execute(candle, state, portfolio)
        state = calculate_metrics.calculate_candle_metrics(candle, state, portfolio)
        print_backtesting_results.print_results(candle, data, state, portfolio)
        update_state.update(state)
        update_portfolio.update(portfolio)
        plt.plot(equity_curve)
        plt.title('Equity Curve')
        plt.ylabel('Equity')
        plt.xlabel('Time')
        plt.savefig(get_absolute_path.absolute(config['paths']['report_directory']) / 'equity_charts/' / f'equity_curve_backtest_{label}.png', dpi=300,
                    bbox_inches='tight')
        plt.clf()
        plt.close()

def backtest_on_val():
    true_data = read_file.read_merged_val_data(False)
    field_of_view = read_file.read_merged_val_data(True)
    predict_position.assign_field_of_view(field_of_view)
    backtest(true_data, label='val')

def backtest_on_test():
    true_data = read_file.read_merged_test_data(False)
    field_of_view = read_file.read_merged_test_data(True)
    predict_position.assign_field_of_view(field_of_view)
    backtest(true_data, label='test')

def backtest_on_train():
    true_data = read_file.read_merged_training_data(False)
    field_of_view = read_file.read_merged_training_data(True)
    predict_position.assign_field_of_view(field_of_view)
    backtest(true_data, label='test')