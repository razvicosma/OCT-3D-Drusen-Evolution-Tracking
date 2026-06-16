APP_STYLE = """
QMainWindow, QDialog {
    background-color: #111827;
}

QWidget {
    background-color: #111827;
    color: #f3f4f6;
    font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu", sans-serif;
    font-size: 14px;
}

QMenuBar {
    background-color: #111827;
    color: #9ca3af;
    border-bottom: 1px solid #1f2937;
    padding: 2px;
}

QMenuBar::item:selected {
    background-color: #1f2937;
    color: #f3f4f6;
    border-radius: 4px;
}

QMenu {
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 20px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #374151;
}

QPushButton {
    background-color: #1f2937;
    color: #f3f4f6;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px 16px;
}

QPushButton:hover {
    background-color: #374151;
    border-color: #4b5563;
}

QPushButton:pressed {
    background-color: #4b5563;
}

QPushButton#bigButton {
    background-color: #1e3a5f;
    border: 1px solid #2563eb;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    min-width: 120px;
    max-width: 140px;
    min-height: 80px;
    max-height: 88px;
    padding: 10px 14px;
}

QPushButton#bigButton:hover {
    background-color: #1d4ed8;
    border-color: #3b82f6;
}

QPushButton#bigButton:pressed {
    background-color: #1e40af;
}

QPushButton#cancelButton {
    background-color: transparent;
    color: #6b7280;
    border: 1px solid #374151;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 13px;
}

QPushButton#cancelButton:hover {
    color: #ef4444;
    border-color: #ef4444;
}

QProgressBar {
    border: none;
    border-radius: 3px;
    background-color: #1f2937;
    height: 6px;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #2563eb;
    border-radius: 3px;
}

QCheckBox, QRadioButton {
    spacing: 8px;
    color: #f3f4f6;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #374151;
    border-radius: 4px;
    background-color: #1f2937;
}

QCheckBox::indicator:checked {
    background-color: #2563eb;
    border-color: #2563eb;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #374151;
    border-radius: 8px;
    background-color: #1f2937;
}

QRadioButton::indicator:checked {
    background-color: #2563eb;
    border-color: #2563eb;
}

QGroupBox {
    border: 1px solid #374151;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 8px;
    font-weight: 600;
    color: #9ca3af;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
}

QListWidget {
    background-color: #1f2937;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 4px;
    outline: none;
}

QListWidget::item {
    padding: 10px 12px;
    border-radius: 6px;
    color: #d1d5db;
}

QListWidget::item:hover {
    background-color: #374151;
    color: #f3f4f6;
}

QListWidget::item:selected {
    background-color: #1e3a5f;
    color: #f3f4f6;
}

QLabel#appTitle {
    font-size: 26px;
    font-weight: 700;
    color: #f3f4f6;
    letter-spacing: 1px;
}

QLabel#appSubtitle {
    font-size: 13px;
    color: #6b7280;
}

QLabel#sectionHeader {
    font-size: 11px;
    font-weight: 600;
    color: #4b5563;
    letter-spacing: 2px;
}

QLabel#loadingTitle {
    font-size: 20px;
    font-weight: 600;
    color: #f3f4f6;
}

QLabel#loadingSubtitle {
    font-size: 13px;
    color: #6b7280;
}
"""
