import talib as ta
import pandas as pd
from src.utils import get_config, read_file, get_absolute_path, check_dir

config = get_config.read_yaml()

def create_features(type = 'training'):
    print(f'Creating features for {type}...')
    data_dir = get_absolute_path.absolute(config['paths'][f'featured_{type}_data_directory'])
    symbols = config['data']['symbols']
    for symbol in symbols:
        data = pd.DataFrame()
        if type == 'training':
            data = read_file.read_raw_training_data(symbol)
        elif type == 'val':
            data = read_file.read_raw_val_data(symbol)
        elif type == 'test':
            data = read_file.read_raw_test_data(symbol)

        close = data['close']
        high = data['high']
        low = data['low']
        openn = data['open']
        volume = data['volume']

        # momentum indicators
        data['rsi'] = ta.RSI(close, timeperiod=14)
        data['mfi'] = ta.MFI(high, low, close, volume.astype(float), timeperiod=14)
        data['adx'] = ta.ADX(high, low, close, timeperiod=14)
        data['bop'] = ta.BOP(open=data['open'], high=high, low=low, close=close)
        slowk, slowd = ta.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3,
                                slowd_matype=0)
        data['stochastic_oscillator-slow'] = slowd
        macd, macdsignal, macdhist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        data['macd-line'] = macd
        data['macd-hist'] = macdhist

        # trend indicators
        data['sma-200'] = ta.SMA(data['close'], timeperiod=200)
        data['ema-50'] = ta.EMA(data['close'], timeperiod=50)
        data['tema-50'] = ta.TEMA(close, timeperiod=50)

        # volume indicators
        data['obv'] = ta.OBV(close, volume.astype(float))

        # volatility indicators
        data['atr'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)

        # other indicators
        data['candle'] = (close - openn) / (high - low)

        symbol = symbol.split('/')[0]
        data.to_csv(get_absolute_path.join_path(data_dir, symbol, 'csv'))