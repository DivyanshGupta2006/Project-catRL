import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from src.utils import get_config, get_absolute_path, check_dir, read_file

config = get_config.read_yaml()

def preprocess(type='training', to_normalize=True):
    print("Preprocessing data....")
    symbols = config['data']['symbols']
    if type == 'training':
        pre_dir = get_absolute_path.absolute(config['paths']['preprocessor_directory'])
        data_dir = get_absolute_path.absolute(config['paths']['processed_training_data_directory'])
        check_dir.check(pre_dir)
        check_dir.check(data_dir)
    elif type == 'val':
        data_dir = get_absolute_path.absolute(config['paths']['processed_val_data_directory'])
        check_dir.check(data_dir)
    elif type == 'test':
        data_dir = get_absolute_path.absolute(config['paths']['processed_test_data_directory'])
        check_dir.check(data_dir)
    for symbol in symbols:
        if type == 'training':
            data = read_file.read_featured_training_data(symbol)
            data = data[data.index >= config['data']['begin_train_date']].copy()
            data_dir = get_absolute_path.absolute(config['paths']['processed_training_data_directory'])
        elif type == 'val':
            data = read_file.read_featured_val_data(symbol)
            data = data[data.index >= config['data']['begin_val_date']].copy()
            data_dir = get_absolute_path.absolute(config['paths']['processed_val_data_directory'])
        elif type == 'test':
            data = read_file.read_featured_test_data(symbol)
            data = data[data.index >= config['data']['begin_test_date']].copy()
            data_dir = get_absolute_path.absolute(config['paths']['processed_test_data_directory'])

        # Handling NaN values
        data = data.ffill()

        if to_normalize:
            # Scaling the data
            if (type == 'training'):
                scaler = MinMaxScaler()
                scaler.fit(data)
                symbol = symbol.split('/')[0]
                joblib.dump(scaler, pre_dir / f'{symbol}.joblib')
            else:
                scaler = read_file.read_preprocessor(symbol)
            data_scaled = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
        else:
            data_scaled = data
        symbol = symbol.split('/')[0]
        path = f'{symbol}.csv'
        data_scaled.to_csv(data_dir / path)
