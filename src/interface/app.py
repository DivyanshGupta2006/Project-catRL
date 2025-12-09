import sys
import os
from PyQt6.QtWidgets import QApplication

from src.interface.theme import apply_dark_theme
from src.interface.window import TradingDashboard
from src.utils import get_config, get_absolute_path

configg = get_config.read_yaml()

def start():
    save_dir = get_absolute_path.absolute(configg['paths']['report_directory'] + 'experiments/')

    app = QApplication(sys.argv)

    apply_dark_theme(app)

    window = TradingDashboard(save_dir)
    window.show()

    sys.exit(app.exec())