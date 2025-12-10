import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLineEdit, QPushButton, QTextEdit, QVBoxLayout
from PyQt6.QtGui import QIcon
from PyQt6 import uic

from src.utils import get_config, get_absolute_path

config = get_config.read_yaml()

asset_dir = get_absolute_path.absolute(config['paths']['asset_directory'])

class App(QWidget):
    def __init__(self):
        super().__init__()
        # self.setWindowTitle('Less Go! Stay Strong!')
        uic.loadUi(asset_dir / 'interface.ui', self)
        # self.setWindowIcon(QIcon('icon.ico'))
        # self.setGeometry(25, 100, 1500, 750)
        #
        # layout = QVBoxLayout()
        # self.setLayout(layout)
        #
        # self.input = QLineEdit()
        # self.input.setPlaceholderText('Enter your Name...')
        # layout.addWidget(self.input)
        #
        # button = QPushButton('I can do it!', clicked=self.sayHello)
        # layout.addWidget(button)
        #
        # self.output = QTextEdit()
        # layout.addWidget(self.output)

    def sayHello(self):
        input = self.input.text()
        res = f'Yes, you can {input}!'
        self.output.setText(res)

app = QApplication(sys.argv)
app.setStyleSheet('''
    QWidget {
        font-size: 25px;
    }
    QPushButton {
        font-size: 20px;
    }
''')
window = App()
window.show()
sys.exit(app.exec())
