import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtester import data_handler, place_order, execute_order, execute_SL_TP
from src.agent import predict_position
from src.risk_management import slippage, stop_loss, take_profit
from src.position_sizing import amount_calculator, portfolio_calculator
from src.update_files import update_state, update_portfolio
from src.utils import read_file, get_config

config = get_config.read_yaml()

def backtest_step(data):
    candle = data_handler.fetch_data(data)
    # execute_SL_TP.execute(candle)
    candle = predict_position.predict(candle)
    Pt = portfolio_calculator.calculate(candle)
    candle = slippage.get_order_price(candle, Pt)
    candle = amount_calculator.calculate(candle, Pt)
    candle = stop_loss.get_stop_loss(candle)
    candle = take_profit.get_take_profit(candle)
    order = place_order.place(candle)
    execute_order.execute(order)

def backtest():
    data = read_file.read_merged_test_data()
    update_state.set_state(data, 100000)
    update_portfolio.set_portfolio()
    for _ in tqdm(range(len(data) - 1), desc="Backtesting Progress"):
        backtest_step(data)
    # last row
    candle = data_handler.fetch_data(data)
    execute_SL_TP.execute(candle)
    print(portfolio_calculator.calculate(candle))
