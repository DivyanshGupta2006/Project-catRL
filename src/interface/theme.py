from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtCore import Qt


def apply_dark_theme(app):
    """
    Applies a high-end 'Obsidian & Emerald' financial terminal theme.
    """
    app.setStyle("Fusion")

    # 1. THE PALETTE (The Foundation)
    palette = QPalette()

    # Backgrounds: Deep, matte darks (reduces eye strain)
    obsidian_bg = QColor(18, 18, 18)  # Main Window
    surface_bg = QColor(30, 30, 30)  # Cards/Panels

    # Text: Off-white (pure white is too harsh on dark backgrounds)
    text_main = QColor(240, 240, 240)
    text_dim = QColor(160, 160, 160)

    # Accents: The "CatRL" Identity
    primary = QColor(46, 139, 87)  # SeaGreen (Your Brand)
    highlight = QColor(0, 230, 118)  # Neon Green (For active states)
    link = QColor(66, 165, 245)  # Soft Blue for hyperlinks
    error = QColor(207, 102, 121)  # Soft Red

    # Apply to Roles
    palette.setColor(QPalette.ColorRole.Window, obsidian_bg)
    palette.setColor(QPalette.ColorRole.WindowText, text_main)
    palette.setColor(QPalette.ColorRole.Base, surface_bg)
    palette.setColor(QPalette.ColorRole.AlternateBase, obsidian_bg)
    palette.setColor(QPalette.ColorRole.ToolTipBase, surface_bg)
    palette.setColor(QPalette.ColorRole.ToolTipText, text_main)
    palette.setColor(QPalette.ColorRole.Text, text_main)
    palette.setColor(QPalette.ColorRole.Button, surface_bg)
    palette.setColor(QPalette.ColorRole.ButtonText, text_main)
    palette.setColor(QPalette.ColorRole.BrightText, error)
    palette.setColor(QPalette.ColorRole.Link, link)
    palette.setColor(QPalette.ColorRole.Highlight, primary)
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)

    app.setPalette(palette)

    # 2. GLOBAL FONTS
    # Prefer modern system fonts
    font = QFont("Segoe UI")
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    app.setFont(font)

    # 3. GLOBAL WIDGET STYLING (Scrollbars, Tooltips, Menus)
    # This injects styles for widgets that are hard to style individually
    app.setStyleSheet("""
        QToolTip {
            background-color: #2b2b2b;
            color: #f0f0f0;
            border: 1px solid #444;
            padding: 5px;
            font-size: 12px;
        }

        /* MODERN SCROLLBARS (Slim & Dark) */
        QScrollBar:vertical {
            border: none;
            background: #121212;
            width: 10px;
            margin: 0px 0px 0px 0px;
        }
        QScrollBar::handle:vertical {
            background: #444;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #2e8b57;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        QScrollBar:horizontal {
            border: none;
            background: #121212;
            height: 10px;
        }
        QScrollBar::handle:horizontal {
            background: #444;
            min-width: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #2e8b57;
        }

        /* TABS (Global fallback) */
        QTabWidget::pane {
            border: 1px solid #333;
            background: #1e1e1e;
        }
    """)


# --- EXPORTED COMPONENT STYLES ---

PROGRESS_BAR_STYLE = """
    QProgressBar {
        border: none;
        background-color: #1a1a1a;
        color: white;
        font-weight: bold;
        text-align: center;
        border-radius: 4px;
    }
    QProgressBar::chunk {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2e8b57, stop:1 #3cb371);
        border-radius: 4px;
    }
"""

INPUT_STYLE = """
    QGroupBox {
        border: 1px solid #333;
        border-radius: 8px;
        margin-top: 20px;
        font-weight: bold;
        color: #2e8b57;  /* Title Color */
        background-color: #1e1e1e;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        left: 15px;
    }

    /* INPUT FIELDS */
    QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
        background-color: #252525;
        color: #f0f0f0;
        border: 1px solid #444;
        padding: 8px;
        border-radius: 4px;
        font-family: 'Consolas', 'Segoe UI Mono', monospace;
    }

    /* FOCUS STATES (The Glow Effect) */
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {
        border: 1px solid #2e8b57;
        background-color: #2a2a2a;
    }

    /* CHECKBOXES */
    QCheckBox {
        spacing: 8px;
        color: #ddd;
    }
    QCheckBox::indicator {
        width: 18px;
        height: 18px;
        border-radius: 3px;
        border: 1px solid #555;
        background: #252525;
    }
    QCheckBox::indicator:checked {
        background-color: #2e8b57;
        border: 1px solid #2e8b57;
        image: url(none); /* You can add a checkmark icon here if you have one */
    }
    QCheckBox::indicator:hover {
        border: 1px solid #2e8b57;
    }
"""

TABLE_STYLE = """
    QTableWidget {
        background-color: #1e1e1e;
        alternate-background-color: #242424; /* Subtle Zebra Striping */
        gridline-color: #333;
        color: #f0f0f0;
        border: 1px solid #333;
        font-size: 13px;
    }

    /* HEADER */
    QHeaderView::section {
        background-color: #181818;
        color: #2e8b57;
        padding: 8px;
        border: none;
        border-bottom: 2px solid #2e8b57;
        font-weight: bold;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 1px;
    }

    /* SELECTION */
    QTableWidget::item:selected {
        background-color: #2e8b57;
        color: white;
    }
    QTableWidget::item:hover {
        background-color: #2a2a2a;
    }
"""