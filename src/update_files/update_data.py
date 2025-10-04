from src.data_processing import download_data, preprocess_data, split_data, feature_engineer, link_data

def update():
    download_data.download()
    split_data.split()
    link_data.link('training-val')
    link_data.link('val-test')
    feature_engineer.create_features(type='training')
    feature_engineer.create_features(type='val')
    feature_engineer.create_features(type='test')
    preprocess_data.preprocess()