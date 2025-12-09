import sys
import os
from PyQt6.QtWidgets import QApplication

# Local imports
# We assume these files exist and are correct as per your instructions
from src.interface.theme import apply_dark_theme
from src.interface.window import TradingDashboard
from src.utils import get_config, get_absolute_path, check_dir


def start():
    """
    Initializes and launches the PyQt6 Trading Dashboard.
    """
    # 1. Load Configuration
    # We load it here so we can catch errors if the yaml file is missing or malformed
    try:
        config = get_config.read_yaml()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load configuration. {e}")
        return

    # 2. Setup Directories
    # Use os.path.join for safe path construction across different operating systems
    report_dir_key = config['paths']['report_directory']

    # Construct the full path: report_directory/experiments/
    experiments_path = report_dir_key + 'experiments/'

    # Get absolute path using your utility
    save_dir = get_absolute_path.absolute(experiments_path)
    check_dir.check(save_dir)

    # 3. Initialize QApplication
    # Check if an instance already exists to prevent 'QApplication created before...' errors
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # 4. Apply Visual Theme
    # (We will make this function much better in the next step: theme.py)
    apply_dark_theme(app)

    # 5. Launch Main Window
    # We pass the save_dir so the window knows where to save/load sessions
    window = TradingDashboard(save_dir)

    # Set a professional title and icon here if needed,
    # though we can also do it inside the Window class.
    window.setWindowTitle("CatRL - Algorithmic Trading Environment")

    window.show()

    # 6. Start Event Loop
    sys.exit(app.exec())