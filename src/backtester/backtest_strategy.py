from tqdm import tqdm
import matplotlib.pyplot as plt

from src.backtester import data_handler, place_order, execute_order, execute_SL_TP, print_backtesting_results, calculate_metrics
from src.strategy import predict_position
from src.risk_management import slippage, stop_loss, take_profit
from src.position_sizing import amount_calculator, portfolio_calculator
from src.update_files import update_state, update_portfolio
from src.utils import read_file, get_config, get_absolute_path

config = get_config.read_yaml()

def backtest_step(data, equity_curve):
    candle = data_handler.fetch_data(data)
    _ = execute_SL_TP.execute(candle)
    calculate_metrics.calculate_candle_metrics(candle)
    candle = predict_position.predict(candle)
    Pt = portfolio_calculator.calculate(candle)
    equity_curve.append(Pt)
    candle = slippage.get_order_price(candle, Pt)
    candle = amount_calculator.calculate(candle, Pt)
    candle = stop_loss.get_stop_loss(candle)
    candle = take_profit.get_take_profit(candle)
    order = place_order.place(candle)
    execute_order.execute(order)
    calculate_metrics.calculate_order_metrics(order)

def backtest():
    data = read_file.read_merged_test_data(False)
    update_state.set_state(config['strategy']['capital'])
    update_portfolio.set_portfolio()
    equity_curve = []
    for _ in tqdm(range(len(data) - 1), desc="Backtesting Progress"):
        backtest_step(data, equity_curve)

    # last row
    candle = data_handler.fetch_data(data)
    execute_SL_TP.execute(candle)
    calculate_metrics.calculate_candle_metrics(candle)
    print_backtesting_results.print_results(candle, data)
    plt.plot(equity_curve)
    plt.title('Equity Curve')
    plt.ylabel('Equity')
    plt.xlabel('Time')
    plt.savefig(get_absolute_path.absolute(config['paths']['report_directory']) / 'equity-charts/' / f'equity_curve_backtest.png', dpi=300,
                bbox_inches='tight')
    plt.close()
