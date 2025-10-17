import numpy as np
import pandas as pd
from tqdm import tqdm

from src.backtester import data_handler, place_order, execute_order, rebalance_state_and_portfolio, execute_SL_TP
from src.agent import predict
from src.risk_management import slippage, stop_loss, take_profit
from src.position_sizing import amount_calculator, portfolio_calculator
from src.update_files import update_state
from src.utils import read_file, get_config

config = get_config.read_yaml()

def backtest_step(data):
    candle = data_handler.fetch_data(data)
    candle = predict.predict_position(candle)
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

    data.index = pd.to_datetime(data.index)
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')

    data.columns = pd.MultiIndex.from_tuples([eval(col) for col in data.columns])
    new_features = ['fiducia','order_price','amt','stop_price','take_price','stop_loss_portion','take_profit_portion']
    for crypto in [x.split('/')[0] for x in config['data']['symbols']]:
        for feature in new_features:
            if (feature, crypto) not in data.columns:
                data[(feature, crypto)] = np.nan

    state = read_file.read_state()
    if state is None:
        state = {}
    state['timestep'] = data.index[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    state['cash'] = 100000
    update_state.update(state)

    for d in tqdm(range(len(data)), desc="Backtesting Progress"):
        backtest_step(data)
