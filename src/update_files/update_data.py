from src.data_processing import download_data, preprocess_data, split_data, feature_engineer

def update():
    download_data.download()
    split_data.split()
    feature_engineer.create_features()
    preprocess_data.preprocess()