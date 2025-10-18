import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtester import data_handler, place_order, execute_order, rebalance_state_and_portfolio, execute_SL_TP
from src.agent import predict_position
from src.risk_management import slippage, stop_loss, take_profit
from src.position_sizing import amount_calculator, portfolio_calculator
from src.update_files import update_state, update_portfolio
from src.utils import read_file, get_config

config = get_config.read_yaml()

def backtest_step(data):
    candle = data_handler.fetch_data(data)
    candle = predict_position.predict(candle)
    execute_SL_TP.execute(candle)
    Pt = portfolio_calculator.calculate(candle)
    candle = slippage.get_order_price(candle, Pt)
    candle = amount_calculator.calculate(candle, Pt)
    candle = stop_loss.get_stop_loss(candle)
    candle = take_profit.get_take_profit(candle)
    order = place_order.place(candle)
    execute_order.execute(order)
    rebalance_state_and_portfolio.rebalance(candle, order)

def backtest():
    data = read_file.read_merged_test_data()
    update_state.set_state(data, 100000)
    state = read_file.read_state()
    update_portfolio.set_portfolio()
    portfolio = read_file.read_portfolio()
    for d in tqdm(range(len(data) - 1), desc="Backtesting Progress"):
        backtest_step(data)
    # TODO: handle last row
