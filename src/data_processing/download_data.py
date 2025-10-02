import ccxt
import time
import pandas as pd
from src.utils import get_absolute_path, get_config, check_dir

config = get_config.read_yaml()

def download():
    symbols = config['data']['symbols']
    data_dir = get_absolute_path.absolute(config['paths']['raw_data_directory'])
    check_dir.check(data_dir)
    try:
        exchange = getattr(ccxt, config['data']['exchange'])()
    except AttributeError:
        print(f"Error: The exchange '{config['data']['exchange']}' is not supported by ccxt.")
    for symbol in symbols:
        all_ohlcv = []
        start = exchange.parse8601(f'{config['data']['start_date']}T00:00:00Z')
        end = exchange.parse8601(f'{config['data']['end_date']}T23:59:59Z')
        while start < end:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, config['data']['timeframe'], start)
                if not ohlcv:
                    break
                all_ohlcv.extend(ohlcv)
                start = ohlcv[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)
            except ccxt.NetworkError as e:
                print(f"Network error: {e}. Retrying...")
                continue
            except ccxt.ExchangeError as e:
                print(f"Exchange error: {e}")
        all_ohlcv = [candle for candle in all_ohlcv if candle[0] <= end]
        data = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        symbol = symbol.split('/')[0]
        path = f'{symbol}.csv'
        print(f'Downloading {symbol}...')
        data.to_csv(data_dir / path)