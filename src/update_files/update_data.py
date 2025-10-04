from src.data_processing import download_data, preprocess_data, split_data, feature_engineer

def update():
    download_data.download()
    split_data.split()
    feature_engineer.create_features(type='training')
    feature_engineer.create_features(type='val')
    feature_engineer.create_features(type='test')
    preprocess_data.preprocess()