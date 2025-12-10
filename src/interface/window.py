import os
import shutil
import json
import datetime
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QLabel, QCheckBox,
                             QGroupBox, QLineEdit, QTabWidget, QScrollArea,
                             QSplitter, QFileDialog, QMessageBox, QFrame,
                             QProgressBar, QFormLayout, QDoubleSpinBox, QSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QFont, QColor, QIcon

# Import Worker from your local structure
from src.interface.workers import Worker
# Import Theme constants
from src.interface.theme import PROGRESS_BAR_STYLE, INPUT_STYLE, TABLE_STYLE

DEFAULT_SAVE_DIR = ""


# --- 1. LIVE TRADING TAB ---
class TradeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header
        header = QLabel("LIVE MARKET OVERVIEW")
        header.setStyleSheet("font-size: 22px; font-weight: bold; color: #ccc;")
        layout.addWidget(header)

        # Splitter for Chart and Details
        splitter = QSplitter(Qt.Orientation.Vertical)

        # A. Chart Placeholder (The "View")
        chart_frame = QFrame()
        chart_frame.setFrameShape(QFrame.Shape.StyledPanel)
        chart_frame.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333; border-radius: 8px;")
        chart_layout = QVBoxLayout(chart_frame)

        lbl_chart = QLabel("[ REAL-TIME CANDLESTICK CHART MODULE ]\n(Waiting for Data Feed...)")
        lbl_chart.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_chart.setStyleSheet("color: #555; font-size: 16px; font-weight: bold;")
        chart_layout.addWidget(lbl_chart)

        # B. Active Positions Table
        table_frame = QFrame()
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(0, 0, 0, 0)

        lbl_table = QLabel("ACTIVE POSITIONS")
        lbl_table.setStyleSheet("font-size: 14px; font-weight: bold; color: #2e8b57; margin-bottom: 5px;")

        self.positions_table = QTableWidget(0, 5)
        self.positions_table.setHorizontalHeaderLabels(["Asset", "Side", "Entry Price", "Mark Price", "PnL %"])
        self.positions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.positions_table.setStyleSheet(TABLE_STYLE)
        self.positions_table.verticalHeader().setVisible(False)

        # Add dummy row for visual confirmation
        self.add_dummy_row("ETH/USDT", "LONG", "3,450.00", "3,510.00", "+1.74%")
        self.add_dummy_row("BTC/USDT", "SHORT", "98,200.00", "98,150.00", "+0.05%")

        table_layout.addWidget(lbl_table)
        table_layout.addWidget(self.positions_table)

        splitter.addWidget(chart_frame)
        splitter.addWidget(table_frame)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def add_dummy_row(self, asset, side, entry, curr, pnl):
        row = self.positions_table.rowCount()
        self.positions_table.insertRow(row)
        self.positions_table.setItem(row, 0, QTableWidgetItem(asset))
        self.positions_table.setItem(row, 1, QTableWidgetItem(side))
        self.positions_table.setItem(row, 2, QTableWidgetItem(entry))
        self.positions_table.setItem(row, 3, QTableWidgetItem(curr))

        item_pnl = QTableWidgetItem(pnl)
        # Green for positive, Red for negative
        item_pnl.setForeground(QColor("#00ff00") if "+" in pnl else QColor("#ff0000"))
        self.positions_table.setItem(row, 4, item_pnl)


# --- 2. SETTINGS TAB ---
class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        lbl_title = QLabel("CONFIGURATION")
        lbl_title.setStyleSheet("font-size: 22px; font-weight: bold; color: #ccc; margin-bottom: 10px;")
        layout.addWidget(lbl_title)

        # Scroll Area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("border: none; background: transparent;")

        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        form_layout = QVBoxLayout(content_widget)
        form_layout.setSpacing(25)

        # Group 1: Model
        group_model = QGroupBox("Model Hyperparameters")
        group_model.setStyleSheet(INPUT_STYLE)
        form_model = QFormLayout()
        form_model.setVerticalSpacing(15)

        self.sb_lr = QDoubleSpinBox()
        self.sb_lr.setRange(0.00001, 1.0)
        self.sb_lr.setSingleStep(0.0001)
        self.sb_lr.setDecimals(5)
        self.sb_lr.setValue(0.0003)
        self.sb_lr.setToolTip("Step size for the optimizer.")

        self.sb_gamma = QDoubleSpinBox()
        self.sb_gamma.setRange(0.1, 1.0)
        self.sb_gamma.setValue(0.99)
        self.sb_gamma.setToolTip("Discount factor for future rewards.")

        self.sb_batch = QSpinBox()
        self.sb_batch.setRange(16, 4096)
        self.sb_batch.setValue(64)

        form_model.addRow("Learning Rate:", self.sb_lr)
        form_model.addRow("Gamma (Discount):", self.sb_gamma)
        form_model.addRow("Batch Size:", self.sb_batch)
        group_model.setLayout(form_model)

        # Group 2: Environment
        group_env = QGroupBox("Trading Environment")
        group_env.setStyleSheet(INPUT_STYLE)
        form_env = QFormLayout()
        form_env.setVerticalSpacing(15)

        self.sb_window = QSpinBox()
        self.sb_window.setRange(10, 500)
        self.sb_window.setValue(30)
        self.sb_window.setToolTip("Lookback period for the agent.")

        self.sb_balance = QDoubleSpinBox()
        self.sb_balance.setRange(100, 10_000_000)
        self.sb_balance.setValue(10000)
        self.sb_balance.setPrefix("$ ")

        form_env.addRow("Window Size (Bars):", self.sb_window)
        form_env.addRow("Initial Capital:", self.sb_balance)
        group_env.setLayout(form_env)

        form_layout.addWidget(group_model)
        form_layout.addWidget(group_env)
        form_layout.addStretch()

        scroll.setWidget(content_widget)
        layout.addWidget(scroll)

        # Reset Button
        btn_reset = QPushButton("Reset to Defaults")
        btn_reset.setFixedWidth(150)
        btn_reset.setStyleSheet(
            "background-color: #333; color: white; border: 1px solid #555; border-radius: 4px; padding: 6px;")
        btn_reset.clicked.connect(self.reset_defaults)
        layout.addWidget(btn_reset, alignment=Qt.AlignmentFlag.AlignRight)

        self.setLayout(layout)

    def reset_defaults(self):
        self.sb_lr.setValue(0.0003)
        self.sb_gamma.setValue(0.99)
        self.sb_batch.setValue(64)
        self.sb_window.setValue(30)
        self.sb_balance.setValue(10000)

    def get_params(self):
        return {
            "learning_rate": self.sb_lr.value(),
            "gamma": self.sb_gamma.value(),
            "batch_size": self.sb_batch.value(),
            "window_size": self.sb_window.value(),
            "initial_balance": self.sb_balance.value()
        }

    def set_params(self, params):
        if "learning_rate" in params: self.sb_lr.setValue(params["learning_rate"])
        if "gamma" in params: self.sb_gamma.setValue(params["gamma"])
        if "batch_size" in params: self.sb_batch.setValue(params["batch_size"])
        if "window_size" in params: self.sb_window.setValue(params["window_size"])
        if "initial_balance" in params: self.sb_balance.setValue(params["initial_balance"])


# --- 3. EXPERIMENT TAB ---
class ExperimentTab(QWidget):
    def __init__(self, settings_tab_ref):
        super().__init__()
        self.settings_tab = settings_tab_ref
        self.generated_images = []
        self.worker = None
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- LEFT PANEL (Controls) ---
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
        left_panel.setStyleSheet("background-color: #1e1e1e; border-right: 1px solid #333;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15)

        # Meta Data
        meta_group = QGroupBox("Experiment Details")
        meta_group.setStyleSheet(INPUT_STYLE)
        meta_layout = QVBoxLayout()

        self.input_title = QLineEdit()
        self.input_title.setPlaceholderText("e.g. DQN_BTC_1h_v1")

        self.input_desc = QTextEdit()
        self.input_desc.setPlaceholderText("Experiment notes...")
        self.input_desc.setMaximumHeight(80)

        meta_layout.addWidget(QLabel("Title:"))
        meta_layout.addWidget(self.input_title)
        meta_layout.addWidget(QLabel("Description:"))
        meta_layout.addWidget(self.input_desc)
        meta_group.setLayout(meta_layout)

        # Execution Flags
        self.controls_group = QGroupBox("Execution Pipeline")
        self.controls_group.setStyleSheet(INPUT_STYLE)
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(10)

        self.cb_update = QCheckBox("1. Update Data")
        self.cb_train = QCheckBox("2. Train Agent")
        self.cb_bt_val = QCheckBox("3. Backtest: Validation")
        self.cb_bt_test = QCheckBox("4. Backtest: Test")

        self.cb_train.toggled.connect(self.handle_train_toggle)

        controls_layout.addWidget(self.cb_update)
        controls_layout.addWidget(self.cb_train)
        controls_layout.addWidget(self.cb_bt_val)
        controls_layout.addWidget(self.cb_bt_test)
        self.controls_group.setLayout(controls_layout)

        # Buttons
        self.btn_start = QPushButton("RUN EXPERIMENT")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start.setStyleSheet("""
            QPushButton { background-color: #2e8b57; font-weight: bold; font-size: 14px; border-radius: 4px; color: white; border: none; }
            QPushButton:hover { background-color: #3cb371; }
            QPushButton:disabled { background-color: #444; color: #888; }
        """)
        self.btn_start.clicked.connect(self.start_execution)

        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Save Session")
        self.btn_save.clicked.connect(self.save_experiment)
        self.btn_save.setStyleSheet(
            "background: #333; color: white; border: 1px solid #444; padding: 5px; border-radius: 4px;")

        self.btn_load = QPushButton("Load Session")
        self.btn_load.clicked.connect(self.load_experiment)
        self.btn_load.setStyleSheet(
            "background: #333; color: white; border: 1px solid #444; padding: 5px; border-radius: 4px;")

        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_load)

        left_layout.addWidget(meta_group)
        left_layout.addWidget(self.controls_group)
        left_layout.addStretch()
        left_layout.addWidget(self.btn_start)
        left_layout.addLayout(btn_layout)

        # --- RIGHT PANEL (Output) ---
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        # Tabs for plots
        self.image_tabs = QTabWidget()
        self.image_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; background: #1a1a1a; }
            QTabBar::tab { background: #252525; color: #888; padding: 8px 20px; }
            QTabBar::tab:selected { background: #2e8b57; color: white; }
        """)

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

        self.image_tabs.addTab(self.scroll_train, "Training Metrics")
        self.image_tabs.addTab(self.scroll_backtest, "Backtest Results")

        # Console / Terminal
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        console_header = QHBoxLayout()
        console_label = QLabel("SYSTEM LOGS")
        console_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #555;")
        btn_clear = QPushButton("Clear")
        btn_clear.setFixedSize(60, 20)
        btn_clear.clicked.connect(lambda: self.console.clear())
        btn_clear.setStyleSheet("font-size: 10px; background: #333; border: none; color: white; border-radius: 2px;")

        console_header.addWidget(console_label)
        console_header.addStretch()
        console_header.addWidget(btn_clear)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 10))
        self.console.setStyleSheet("background-color: #0f0f0f; color: #00e676; border: 1px solid #333;")

        self.lbl_progress_details = QLabel("Ready to initialize.")
        self.lbl_progress_details.setStyleSheet("color: #aaa; font-style: italic;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        bottom_layout.addLayout(console_header)
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
        # Disable manual backtest selection if training is selected (auto flow)
        self.cb_bt_val.setDisabled(checked)
        self.cb_bt_test.setDisabled(checked)
        if checked:
            self.cb_bt_val.setChecked(False)
            self.cb_bt_test.setChecked(False)

    def start_execution(self):
        title = self.input_title.text().strip()
        if not title:
            QMessageBox.warning(self, "Input Error", "Please provide a Title for this experiment.")
            return

        self.btn_start.setEnabled(False)
        self.btn_start.setText("RUNNING...")

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
        self.worker.log_signal.connect(self.append_log)
        self.worker.image_signal.connect(self.display_and_track_image)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.progress_text_signal.connect(self.lbl_progress_details.setText)
        self.worker.finished_signal.connect(self.on_execution_finished)

        self.worker.start()

    def append_log(self, text):
        self.console.append(text)
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_execution_finished(self):
        self.console.append("\n--- EXECUTION FINISHED ---")
        self.btn_start.setEnabled(True)
        self.btn_start.setText("RUN EXPERIMENT")
        self.lbl_progress_details.setText("Job Complete.")

    def display_and_track_image(self, category, image_path):
        if not os.path.exists(image_path):
            self.console.append(f"ERR: Image not found at {image_path}")
            return

        self.generated_images.append({'category': category, 'path': image_path})

        container = QFrame()
        container.setStyleSheet(
            "background-color: #222; border-radius: 5px; margin-bottom: 10px; border: 1px solid #444;")
        vbox = QVBoxLayout(container)

        lbl_img = QLabel()
        pixmap = QPixmap(image_path)
        # Resize safely
        scaled_pixmap = pixmap.scaled(QSize(800, 500), Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        lbl_img.setPixmap(scaled_pixmap)
        lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_cap = QLabel(os.path.basename(image_path))
        lbl_cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_cap.setStyleSheet("color: #888; margin-top: 5px;")

        vbox.addWidget(lbl_img)
        vbox.addWidget(lbl_cap)

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
            QMessageBox.warning(self, "Error", "Title required to save.")
            return

        safe_title = "".join([c for c in title if c.isalnum() or c in (' ', '_', '-')]).strip()
        save_path = os.path.join(DEFAULT_SAVE_DIR, safe_title)

        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

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
                dest_path = os.path.join(save_path, fname)
                shutil.copy2(img['path'], dest_path)
                data["images"].append({"category": img['category'], "filename": fname})

            with open(os.path.join(save_path, "session.json"), 'w') as f:
                json.dump(data, f, indent=4)

            QMessageBox.information(self, "Saved", f"Experiment saved successfully to:\n{save_path}")

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save experiment:\n{str(e)}")

    def load_experiment(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Experiment Folder", DEFAULT_SAVE_DIR)
        if not folder: return

        json_path = os.path.join(folder, "session.json")
        if not os.path.exists(json_path):
            QMessageBox.critical(self, "Error", "No session.json found in selected folder.")
            return

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            self.input_title.setText(data.get("title", ""))
            self.input_desc.setText(data.get("description", ""))
            self.console.setText(data.get("logs", ""))
            if "hyperparams" in data:
                self.settings_tab.set_params(data["hyperparams"])

            self.clear_layout(self.layout_train)
            self.clear_layout(self.layout_backtest)
            self.generated_images = []

            for img in data.get("images", []):
                full_path = os.path.join(folder, img["filename"])
                self.display_and_track_image(img["category"], full_path)

            self.console.append(f"\n>> Loaded Experiment: {data.get('title')}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))


# --- 4. MAIN WINDOW (Simplified) ---
class TradingDashboard(QMainWindow):
    def __init__(self, save_dir):
        super().__init__()
        global DEFAULT_SAVE_DIR
        DEFAULT_SAVE_DIR = save_dir

        self.setWindowTitle("CatRL - Algorithmic Trading Environment")
        self.resize(1400, 900)

        # Set Window Icon explicitly
        if os.path.exists("icon.ico"):
            self.setWindowIcon(QIcon("icon.ico"))

        # Initialize Tabs directly (No Stacked Widget needed)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border-top: 2px solid #2e8b57; }
            QTabBar::tab { background: #252525; color: #aaa; padding: 10px 20px; font-weight: bold; }
            QTabBar::tab:selected { background: #2e8b57; color: white; }
        """)

        self.trade_tab = TradeTab()
        self.settings_tab = SettingsTab()
        self.experiment_tab = ExperimentTab(self.settings_tab)

        self.tabs.addTab(self.trade_tab, "Live Trade")
        self.tabs.addTab(self.experiment_tab, "Experiment Lab")
        self.tabs.addTab(self.settings_tab, "Settings")

        # Set Tabs as Central Widget
        self.setCentralWidget(self.tabs)

        # Start at Experiment Tab (index 1) or Live Trade (index 0)
        self.tabs.setCurrentIndex(1)