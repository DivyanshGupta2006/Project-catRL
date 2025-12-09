import os
import shutil
import json
import datetime
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QLabel, QCheckBox,
                             QGroupBox, QLineEdit, QTabWidget, QScrollArea,
                             QSplitter, QFileDialog, QMessageBox, QFrame,
                             QProgressBar, QFormLayout, QDoubleSpinBox, QSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont

from src.interface.workers import Worker
from src.interface.theme import PROGRESS_BAR_STYLE

DEFAULT_SAVE_DIR = ""


# --- TAB 1: TRADE (Coming Soon) ---
class TradeTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel("Live Trading Module\n[COMING SOON]")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 24px; color: #555; font-weight: bold;")

        layout.addWidget(label)
        self.setLayout(layout)


# --- TAB 2: SETTINGS (Hyperparameters) ---
class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()

        # 1. Model Parameters
        group_model = QGroupBox("Model Hyperparameters")
        form_model = QFormLayout()

        self.sb_lr = QDoubleSpinBox()
        self.sb_lr.setRange(0.00001, 1.0)
        self.sb_lr.setSingleStep(0.0001)
        self.sb_lr.setDecimals(5)
        self.sb_lr.setValue(0.0003)

        self.sb_gamma = QDoubleSpinBox()
        self.sb_gamma.setRange(0.1, 1.0)
        self.sb_gamma.setValue(0.99)

        self.sb_batch = QSpinBox()
        self.sb_batch.setRange(16, 2048)
        self.sb_batch.setValue(64)

        form_model.addRow("Learning Rate:", self.sb_lr)
        form_model.addRow("Gamma (Discount):", self.sb_gamma)
        form_model.addRow("Batch Size:", self.sb_batch)
        group_model.setLayout(form_model)

        # 2. Environment / Trade Params
        group_env = QGroupBox("Trading Environment")
        form_env = QFormLayout()

        self.sb_window = QSpinBox()
        self.sb_window.setRange(10, 200)
        self.sb_window.setValue(30)

        self.sb_balance = QDoubleSpinBox()
        self.sb_balance.setRange(100, 1000000)
        self.sb_balance.setValue(10000)

        form_env.addRow("Window Size (Lookback):", self.sb_window)
        form_env.addRow("Initial Balance ($):", self.sb_balance)
        group_env.setLayout(form_env)

        layout.addWidget(group_model)
        layout.addWidget(group_env)
        layout.addStretch()
        self.setLayout(layout)

    def get_params(self):
        """Returns a dict of all current settings."""
        return {
            "learning_rate": self.sb_lr.value(),
            "gamma": self.sb_gamma.value(),
            "batch_size": self.sb_batch.value(),
            "window_size": self.sb_window.value(),
            "initial_balance": self.sb_balance.value()
        }

    def set_params(self, params):
        """Populates fields from a dictionary (used when loading)."""
        if "learning_rate" in params: self.sb_lr.setValue(params["learning_rate"])
        if "gamma" in params: self.sb_gamma.setValue(params["gamma"])
        if "batch_size" in params: self.sb_batch.setValue(params["batch_size"])
        if "window_size" in params: self.sb_window.setValue(params["window_size"])
        if "initial_balance" in params: self.sb_balance.setValue(params["initial_balance"])


# --- TAB 3: EXPERIMENT (Main Execution) ---
class ExperimentTab(QWidget):
    def __init__(self, settings_tab_ref):
        super().__init__()
        self.settings_tab = settings_tab_ref  # Reference to Settings Tab to pull data
        self.generated_images = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # --- LEFT PANEL ---
        left_panel = QFrame()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # Inputs
        meta_group = QGroupBox("Experiment Details")
        meta_layout = QVBoxLayout()
        self.input_title = QLineEdit()
        self.input_title.setPlaceholderText("Experiment Title")
        self.input_desc = QTextEdit()
        self.input_desc.setPlaceholderText("Description...")
        self.input_desc.setMaximumHeight(80)
        meta_layout.addWidget(QLabel("Title:"))
        meta_layout.addWidget(self.input_title)
        meta_layout.addWidget(QLabel("Description:"))
        meta_layout.addWidget(self.input_desc)
        meta_group.setLayout(meta_layout)

        # Config
        self.controls_group = QGroupBox("Execution Flags")
        controls_layout = QVBoxLayout()
        self.cb_update = QCheckBox("Update Data")
        self.cb_train = QCheckBox("Train Agent")
        self.cb_bt_val = QCheckBox("Backtest: Validation")
        self.cb_bt_test = QCheckBox("Backtest: Test")
        self.cb_train.toggled.connect(self.handle_train_toggle)

        controls_layout.addWidget(self.cb_update)
        controls_layout.addWidget(self.cb_train)
        controls_layout.addWidget(self.cb_bt_val)
        controls_layout.addWidget(self.cb_bt_test)
        self.controls_group.setLayout(controls_layout)

        # Buttons
        self.btn_start = QPushButton("RUN EXPERIMENT")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("background-color: #2e8b57; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_execution)

        self.btn_save = QPushButton("Save Experiment")
        self.btn_save.clicked.connect(self.save_experiment)

        self.btn_load = QPushButton("Load Previous Experiment")
        self.btn_load.clicked.connect(self.load_experiment)

        left_layout.addWidget(meta_group)
        left_layout.addWidget(self.controls_group)
        left_layout.addStretch()
        left_layout.addWidget(self.btn_start)
        left_layout.addWidget(self.btn_save)
        left_layout.addWidget(self.btn_load)

        # --- RIGHT PANEL ---
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Tabs
        self.image_tabs = QTabWidget()
        self.tab_train = QWidget()
        self.layout_train = QVBoxLayout(self.tab_train)
        self.scroll_train = QScrollArea()
        self.scroll_train.setWidgetResizable(True)
        self.scroll_train.setWidget(self.tab_train)

        self.tab_backtest = QWidget()
        self.layout_backtest = QVBoxLayout(self.tab_backtest)
        self.scroll_backtest = QScrollArea()
        self.scroll_backtest.setWidgetResizable(True)
        self.scroll_backtest.setWidget(self.tab_backtest)

        self.image_tabs.addTab(self.scroll_train, "Training Trajectories")
        self.image_tabs.addTab(self.scroll_backtest, "Backtest Equity Charts")

        # Console & Progress
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 10))
        self.console.setStyleSheet("background-color: #0c0c0c; color: #00ff00;")

        self.lbl_progress_details = QLabel("Ready")
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setValue(0)

        bottom_layout.addWidget(self.console)
        bottom_layout.addWidget(self.lbl_progress_details)
        bottom_layout.addWidget(self.progress_bar)

        right_splitter.addWidget(self.image_tabs)
        right_splitter.addWidget(bottom_widget)
        right_splitter.setStretchFactor(0, 2)
        right_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_splitter)

    def handle_train_toggle(self, checked):
        self.cb_bt_val.setDisabled(checked)
        self.cb_bt_test.setDisabled(checked)
        if checked:
            self.cb_bt_val.setChecked(False)
            self.cb_bt_test.setChecked(False)

    def start_execution(self):
        title = self.input_title.text().strip()
        if not title:
            QMessageBox.warning(self, "Input Error", "Please provide a Title.")
            return

        # PULL SETTINGS FROM SETTINGS TAB
        hyperparams = self.settings_tab.get_params()

        self.console.clear()
        self.clear_layout(self.layout_train)
        self.clear_layout(self.layout_backtest)
        self.generated_images = []
        self.progress_bar.setValue(0)
        self.console.append(f"Loaded Settings: {json.dumps(hyperparams, indent=2)}")

        config = {
            'flags': {
                'update_data': self.cb_update.isChecked(),
                'train_agent': self.cb_train.isChecked(),
                'backtest_val': self.cb_bt_val.isChecked(),
                'backtest_test': self.cb_bt_test.isChecked()
            },
            'hyperparams': hyperparams
        }

        self.worker = Worker(config)
        self.worker.log_signal.connect(self.console.append)
        self.worker.image_signal.connect(self.display_and_track_image)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.progress_text_signal.connect(self.lbl_progress_details.setText)
        self.worker.finished_signal.connect(lambda: self.console.append("\n--- DONE ---"))
        self.worker.start()

    def display_and_track_image(self, category, image_path):
        if not os.path.exists(image_path):
            self.console.append(f"ERR: Image not found at {image_path}")
            return

        self.generated_images.append({'category': category, 'path': image_path})

        container = QFrame()
        container.setStyleSheet("background-color: #222; border-radius: 5px; margin-bottom: 10px;")
        vbox = QVBoxLayout(container)
        lbl_img = QLabel()
        pixmap = QPixmap(image_path)
        lbl_img.setPixmap(pixmap.scaledToWidth(800, Qt.TransformationMode.SmoothTransformation))
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(lbl_img)
        vbox.addWidget(QLabel(os.path.basename(image_path)))

        if category == 'train':
            self.layout_train.addWidget(container)
        else:
            self.layout_backtest.addWidget(container)

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def save_experiment(self):
        title = self.input_title.text().strip()
        if not title:
            QMessageBox.warning(self, "Error", "Title required.")
            return

        safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '_', '-')]).strip()
        save_path = os.path.join(DEFAULT_SAVE_DIR, safe_title)
        if not os.path.exists(save_path): os.makedirs(save_path)

        # Include Settings in the save file
        current_params = self.settings_tab.get_params()

        data = {
            "title": title,
            "description": self.input_desc.toPlainText(),
            "logs": self.console.toPlainText(),
            "timestamp": str(datetime.datetime.now()),
            "hyperparams": current_params,
            "images": []
        }

        for img in self.generated_images:
            fname = os.path.basename(img['path'])
            shutil.copy2(img['path'], os.path.join(save_path, fname))
            data["images"].append({"category": img['category'], "filename": fname})

        with open(os.path.join(save_path, "session.json"), 'w') as f:
            json.dump(data, f, indent=4)

        QMessageBox.information(self, "Saved", f"Saved to {save_path}")

    def load_experiment(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", DEFAULT_SAVE_DIR)
        if not folder: return

        try:
            with open(os.path.join(folder, "session.json"), 'r') as f:
                data = json.load(f)

            self.input_title.setText(data.get("title", ""))
            self.input_desc.setText(data.get("description", ""))
            self.console.setText(data.get("logs", ""))

            # Restore Settings if they exist
            if "hyperparams" in data:
                self.settings_tab.set_params(data["hyperparams"])
                self.console.append("\n>> Restored Hyperparameters from file.")

            self.clear_layout(self.layout_train)
            self.clear_layout(self.layout_backtest)
            self.generated_images = []

            for img in data.get("images", []):
                self.display_and_track_image(img["category"], os.path.join(folder, img["filename"]))

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# --- MAIN WINDOW (The Container) ---
class TradingDashboard(QMainWindow):
    def __init__(self, save_dir):
        global DEFAULT_SAVE_DIR
        super().__init__()
        DEFAULT_SAVE_DIR = save_dir
        self.setWindowTitle("catRL")
        self.resize(1400, 900)

        # Main Tab Widget
        self.main_tabs = QTabWidget()

        # Initialize Sub-Tabs
        self.trade_tab = TradeTab()
        self.settings_tab = SettingsTab()
        self.experiment_tab = ExperimentTab(self.settings_tab)  # PASSING DEPENDENCY

        # Add to Main Tabs
        # Order: Trade | Experiment | Settings
        self.main_tabs.addTab(self.trade_tab, "Trade")
        self.main_tabs.addTab(self.experiment_tab, "Experiment")
        self.main_tabs.addTab(self.settings_tab, "Settings")

        self.setCentralWidget(self.main_tabs)