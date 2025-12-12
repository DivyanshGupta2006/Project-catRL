import sys

from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow

from src.utils import get_config, get_absolute_path

config = get_config.read_yaml()
asset_dir = get_absolute_path.absolute(config['paths']['asset_directory'])
equity_charts_directory = get_absolute_path.absolute(config['paths']['report_directory'] + 'equity_charts/')
experiments_directory = get_absolute_path.absolute(config['paths']['report_directory'] + 'experiments/')

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(asset_dir / 'interface.ui', self)

app = QApplication(sys.argv)

app.setStyleSheet("""
        /* MAIN WINDOW BACKGROUND */
        QMainWindow, QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
            font-size: 10pt;
        }

        /* MENU BAR */
        QMenuBar {
            background-color: #2d2d2d;
            border-bottom: 1px solid #3d3d3d;
        }
        QMenuBar::item:selected {
            background-color: #3d3d3d;
        }

        /* TABS (Top Main Tabs) */
        QTabWidget::pane {
            border: 1px solid #3d3d3d;
            background-color: #1e1e1e;
        }
        QTabBar::tab {
            background: #2d2d2d;
            padding: 8px 20px;
            color: #b0b0b0;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background: #3e3e3e; /* Lighter grey for active */
            color: #4caf50;      /* Green text for active */
            border-bottom: 2px solid #4caf50;
        }

        /* GROUP BOXES (The containers on the left) */
        QGroupBox {
            border: 1px solid #3d3d3d;
            margin-top: 20px;
            font-weight: bold;
            color: #4caf50; /* Green Title */
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }

        /* INPUT FIELDS */
        QLineEdit, QTextEdit {
            background-color: #2d2d2d;
            border: 1px solid #3d3d3d;
            color: #ffffff;
            padding: 4px;
            border-radius: 3px;
        }
        QLineEdit:focus, QTextEdit:focus {
            border: 1px solid #4caf50; /* Green border when typing */
        }

        /* BUTTONS */
        QPushButton {
            background-color: #3d3d3d;
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #4d4d4d;
        }

        /* THE BIG GREEN BUTTON */
        /* You must set the objectName of your run button to 'btn_run' in Designer for this to work */
        QPushButton#btn_run {
            background-color: #4caf50; 
            font-weight: bold;
        }
        QPushButton#btn_run:hover {
            background-color: #45a049;
        }

        /* CHECKBOXES */
        QCheckBox {
            color: #b0b0b0;
        }
        QCheckBox::indicator:checked {
            background-color: #4caf50;
            border-radius: 2px;
        }
    """)

window = App()
window.show()
sys.exit(app.exec())
