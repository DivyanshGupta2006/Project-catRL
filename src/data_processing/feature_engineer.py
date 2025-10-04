import talib as ta
import pandas as pd
from src.utils import get_config, read_file, get_absolute_path, check_dir

config = get_config.read_yaml()

def create_features(type = 'training'):
    print('Creating features...')
    data_dir = get_absolute_path.absolute(config['paths'][f'featured_{type}_data_directory'])
    check_dir.check(data_dir)
    symbols = config['data']['symbols']
    for symbol in symbols:
        data = pd.DataFrame()
        if type == 'training':
            data = read_file.read_raw_training_data(symbol)
        elif type == 'val':
            data = read_file.read_raw_val_data(symbol)
        elif type == 'test':
            data = read_file.read_raw_test_data(symbol)
        data['rsi'] = ta.RSI(data['close'], timeperiod=14)
        data['sma-50'] = ta.SMA(data['close'], timeperiod=50)
        data['sma-100'] = ta.SMA(data['close'], timeperiod=100)
        data['sma-200'] = ta.SMA(data['close'], timeperiod=200)
        data['ema-50'] = ta.EMA(data['close'], timeperiod=50)
        data['ema-100'] = ta.EMA(data['close'], timeperiod=100)
        data['ema-200'] = ta.EMA(data['close'], timeperiod=200)
        data['atr'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        data['adx'] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        symbol = symbol.split('/')[0]
        path = f'{symbol}.csv'
        data.to_csv(data_dir / path)