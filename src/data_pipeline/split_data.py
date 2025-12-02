from src.utils import get_config, read_file, check_dir, get_absolute_path

config = get_config.read_yaml()

def split():

    symbols = config['data']['symbols']
    print("Splitting the data...")
    train_data_dir = get_absolute_path.absolute(config['paths']['raw_training_data_directory'])
    test_data_dir = get_absolute_path.absolute(config['paths']['raw_test_data_directory'])
    val_data_dir = get_absolute_path.absolute(config['paths']['raw_val_data_directory'])
    check_dir.check(train_data_dir)
    check_dir.check(test_data_dir)
    check_dir.check(val_data_dir)
    for symbol in symbols:
        df = read_file.read_raw_data(symbol)
        df_train = df[df.index <= config['data']['train_val_split']]
        df_val = df[(df.index > config['data']['train_val_split']) & (df.index <= config['data']['val_test_split'])]
        df_test = df[df.index > config['data']['val_test_split']]
        if symbol == 'ETH/USDT':
            print(f"Train set shape: {df_train.shape}")
            print(f"Validation set shape: {df_val.shape}")
            print(f"Test set shape: {df_test.shape}")
        symbol = symbol.split('/')[0]
        path = f'{symbol}.csv'
        df_train.to_csv(train_data_dir / path)
        df_val.to_csv(val_data_dir / path)
        df_test.to_csv(test_data_dir / path)