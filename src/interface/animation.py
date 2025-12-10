import sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QProgressBar,
                             QGraphicsDropShadowEffect, QApplication)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont


class CatRLSplash(QWidget):
    # Signal to let the app know animation is done
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFixedSize(680, 400)

        # 1. Remove Window Frame & Make Background Transparent
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.init_ui()
        self.start_loading_sequence()

    def init_ui(self):
        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 2. Container (The actual visible card)
        self.container = QLabel(self)
        self.container.setStyleSheet("""
            QLabel {
                background-color: #121212;
                border: 1px solid #333;
                border-radius: 20px;
            }
        """)
        # Add Drop Shadow for depth
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 180))
        self.container.setGraphicsEffect(shadow)

        container_layout = QVBoxLayout(self.container)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.setSpacing(5)

        # 3. Branding (Logo)
        lbl_logo = QLabel("CAT<font color='#2e8b57'>RL</font>")
        lbl_logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_logo.setStyleSheet("font-size: 80px; font-weight: 900; color: white; font-family: 'Segoe UI', sans-serif;")

        lbl_subtitle = QLabel("AUTONOMOUS CRYPTO TRADING AGENT")
        lbl_subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_subtitle.setStyleSheet(
            "font-size: 14px; color: #888; letter-spacing: 6px; font-weight: 600; margin-bottom: 30px;")

        # 4. Loading Bar
        self.progress = QProgressBar()
        self.progress.setFixedWidth(500)
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #2e8b57;
                border-radius: 2px;
            }
        """)

        # 5. Loading Text (Updates dynamically)
        self.lbl_loading = QLabel("Initializing Core Systems...")
        self.lbl_loading.setStyleSheet("color: #666; font-size: 10px; font-style: italic; margin-top: 10px;")
        self.lbl_loading.setAlignment(Qt.AlignmentFlag.AlignCenter)

        container_layout.addStretch()
        container_layout.addWidget(lbl_logo)
        container_layout.addWidget(lbl_subtitle)
        container_layout.addSpacing(20)
        container_layout.addWidget(self.progress, alignment=Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.lbl_loading)
        container_layout.addStretch()

        layout.addWidget(self.container)

    def start_loading_sequence(self):
        # Simulation of loading steps
        self.counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(30)  # Speed of animation (lower = faster)

        self.loading_text_stages = [
            (10, "Loading Configuration..."),
            (30, "Initializing Neural Networks..."),
            (50, "Connecting to Market Data Feeds..."),
            (70, "Restoring Session State..."),
            (85, "Optimizing Execution Engine..."),
            (95, "System Ready.")
        ]

    def update_progress(self):
        self.counter += 1
        self.progress.setValue(self.counter)

        # Update text based on progress
        for limit, text in self.loading_text_stages:
            if self.counter == limit:
                self.lbl_loading.setText(text)
                break

        if self.counter >= 100:
            self.timer.stop()
            self.finished.emit()  # Notify App to close splash and open main