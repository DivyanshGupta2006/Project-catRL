from src.data_processing import download_data, preprocess_data, split_data

def update():
    download_data.download()
    split_data.split()
    preprocess_data.preprocess()