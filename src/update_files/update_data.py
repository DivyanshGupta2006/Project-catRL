from src.data_pipeline import download_data, preprocess_data, split_data, feature_engineer, link_data, merge_data

def update():
    a = input('Would you like to download the data? (y/n): ')
    if a.lower() == 'y':
        download_data.download()
    split_data.split()
    link_data.link('training-val')
    link_data.link('val-test')
    
    feature_engineer.create_features(type='training')
    feature_engineer.create_features(type='val')
    feature_engineer.create_features(type='test')
    
    preprocess_data.preprocess(type='training', to_normalize=True)
    preprocess_data.preprocess(type='val', to_normalize=True)
    preprocess_data.preprocess(type='test', to_normalize=True)

    merge_data.merge_normalized(type='training')
    merge_data.merge_normalized(type='val')
    merge_data.merge_normalized(type='test')

    merge_data.merge_unnormalized(type='test')
    merge_data.merge_unnormalized(type='val')
    merge_data.merge_unnormalized(type='training')