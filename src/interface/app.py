import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon

# Imports
from src.interface.theme import apply_dark_theme
from src.interface.window import TradingDashboard
from src.interface.animation import CatRLSplash  # <--- Import new splash
from src.utils import get_config, get_absolute_path

# Global references to keep windows from being garbage collected
window = None
splash = None


def start():
    global window, splash

    # 1. Config Loading
    try:
        config = get_config.read_yaml()
    except:
        config = {}  # Safe fallback

    report_path = config['paths']['report_directory']
    save_dir = get_absolute_path.absolute(report_path + 'experiments/')

    # 2. App Init
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Apply Theme
    apply_dark_theme(app)

    # Set Taskbar Icon
    if os.path.exists('icon.ico'):
        app.setWindowIcon(QIcon('icon.ico'))

    # 3. DEFINE TRANSITION LOGIC
    def show_main_window():
        global window, splash
        # Initialize Main Window
        window = TradingDashboard(save_dir)
        window.show()
        # Close Splash
        if splash:
            splash.close()

    # 4. LAUNCH SPLASH
    splash = CatRLSplash()
    # When splash signals 'finished', run show_main_window
    splash.finished.connect(show_main_window)
    splash.show()

    sys.exit(app.exec())