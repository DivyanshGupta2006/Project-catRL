import torch

from src.backtester import place_order, execute_order, calculate_metrics, execute_SL_TP
from src.position_sizing import fiducia_calculator, portfolio_calculator, amount_calculator
from src.risk_management import slippage, stop_loss, take_profit
from src.utils import convert

class Environment:

    def __init__(self,
                 data,
                 seq_len,
                 capital,
                 symbols):
        self.data = data
        self.seq_len = seq_len
        self.capital = capital
        self.symbols = symbols

        for idx, symbol in enumerate(self.symbols):
            self.symbols[idx] = symbol.split('/')[0]

        self.current_step = self.seq_len
        self.prev_portfolio = self.capital

    def _get_states(self, field_of_view):
        try:
            obs = field_of_view.iloc[self.current_step - self.seq_len : self.current_step].copy()
            states = torch.tensor(obs.values, dtype=torch.float32)
            return states.unsqueeze(0)
        except:
            return None

    def _get_reward(self, prev, new):
        return (new - prev) / prev

    def step(self, raw_action, field_of_view):
        row1 = self.data.iloc[self.current_step - 1]
        row2 = self.data.iloc[self.current_step]
        prev_candle = convert.convert_to_dict(row1)
        candle = convert.convert_to_dict(row2)

        fiduciae = fiducia_calculator.calculate(torch.tensor(raw_action))

        fiduciae = fiduciae.tolist()

        for idx, crypto in enumerate(self.symbols):
            prev_candle[crypto]['fiducia'] = fiduciae[idx]

        prev_candle = slippage.get_order_price(prev_candle, self.prev_portfolio)
        prev_candle = amount_calculator.calculate(prev_candle, self.prev_portfolio)
        prev_candle = stop_loss.get_stop_loss(prev_candle)
        prev_candle = take_profit.get_take_profit(prev_candle)

        order = place_order.place(prev_candle)
        execute_order.execute(order)
        execute_SL_TP.execute(candle)

        new_portfolio = portfolio_calculator.calculate(candle)
        reward = self._get_reward(self.prev_portfolio, new_portfolio)

        self.current_step += 1
        next_states = self._get_states(field_of_view)

        return next_states, reward

    def reset(self, field_of_view):
        self.current_step = self.seq_len
        self.prev_portfolio = self.capital
        return self._get_states(field_of_view)