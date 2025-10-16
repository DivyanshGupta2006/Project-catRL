import json
from src.backtester import data_handler, place_order, execute_order, rebalance
from src.agent import predict
from src.risk_management import slippage, stop_loss, take_profit
from src.position_sizing import amount_calculator, portfolio_calculator
from src.update_files import update_state
from src.utils import read_file

def backtest_step(data):
    candle = data_handler.fetch_data(data)
    candle = predict.predict_position(candle)
    Pt = portfolio_calculator.calculate()
    candle = slippage.get_order_price(candle, Pt)
    candle = amount_calculator.calculate(candle, Pt)
    candle = stop_loss.get_stop_loss(candle)
    candle = take_profit.get_take_profit(candle)
    order = place_order.place(candle)
    execute_order.execute(order)
    rebalance.rebalance_portfolio(order)
    rebalance.rebalance_state(order)

def backtest():
    data = read_file.read_merged_test_data()
    state = read_file.read_state()
    state['timestep'] = data.index[0]
    update_state.update(state)
    for d in range(len(data)):
        backtest_step(data)