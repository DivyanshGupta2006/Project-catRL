from src.data_processing import download_data, preprocess_data, split_data, feature_engineer, link_data, merge_data

def update():
    a = input('Download Data? (y/n): ')
    if a.lower() == 'y':
        download_data.download()
    split_data.split()
    link_data.link('training-val')
    link_data.link('val-test')
    feature_engineer.create_features(type='training')
    feature_engineer.create_features(type='val')
    feature_engineer.create_features(type='test')
    preprocess_data.preprocess(type='training')
    preprocess_data.preprocess(type='val')
    preprocess_data.preprocess(type='test')
    merge_data.merge(type='training')
    merge_data.merge(type='val')
    merge_data.merge(type='test')