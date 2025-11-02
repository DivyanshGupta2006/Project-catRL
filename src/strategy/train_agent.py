import torch

from src.utils import get_config

config = get_config.read_yaml()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    input_dim = len(config['data']['symbols']) * config['data']['num_features']