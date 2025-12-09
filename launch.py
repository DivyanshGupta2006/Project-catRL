import warnings
warnings.filterwarnings('ignore')

from src.interface.app import start
from src import main

if __name__ == '__main__':
    start()
    # main.start()