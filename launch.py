import warnings
warnings.filterwarnings('ignore')

from src.interface import app
from src import main

if __name__ == '__main__':
    # app.run(main.start)
    main.start()