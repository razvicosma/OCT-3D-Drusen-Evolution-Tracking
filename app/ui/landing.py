import os
from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QListWidgetItem, QFileDialog
)

from app.config import MAX_RECENT, PAGE_MARGINS


def _age(ts):

    try:
        delta = datetime.now() - datetime.fromisoformat(ts)
        s = delta.total_seconds()
        if s < 3600:
            return f"{int(s // 60)}m ago"
        if s < 86400:
            return f"{int(s // 3600)}h ago"
        return f"{int(s // 86400)}d ago"
    except Exception:
        return ""


class LandingPage(QWidget):
    open_folder = Signal(str)
    open_volume = Signal(str)

    def __init__(self, recent_list, parent=None):

        super().__init__(parent)
        self.recent_list = recent_list
        self._build()

    def _build(self):

        root = QVBoxLayout(self)
        root.setContentsMargins(*PAGE_MARGINS)
        root.setSpacing(0)

        root.addStretch(1)

        title = QLabel("OCT Drusen Tracker")
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("3D volume reconstruction & retinal layer segmentation")
        subtitle.setObjectName("appSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        root.addWidget(title)
        root.addSpacing(6)
        root.addWidget(subtitle)
        root.addSpacing(36)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)
        btn_row.addStretch()

        self.btn_folder = QPushButton("Select Folder\n\nSparse OCT scans")
        self.btn_folder.setObjectName("bigButton")
        self.btn_folder.setCursor(Qt.PointingHandCursor)
        self.btn_folder.clicked.connect(self._pick_folder)

        self.btn_volume = QPushButton("Select Volume\n\nPre-built .npz file")
        self.btn_volume.setObjectName("bigButton")
        self.btn_volume.setCursor(Qt.PointingHandCursor)
        self.btn_volume.clicked.connect(self._pick_volume)

        btn_row.addWidget(self.btn_folder)
        btn_row.addWidget(self.btn_volume)
        btn_row.addStretch()
        root.addLayout(btn_row)

        root.addSpacing(28)

        center = QHBoxLayout()
        center.addStretch()

        recent_col = QVBoxLayout()
        recent_col.setSpacing(0)

        header = QLabel("RECENT")
        header.setObjectName("sectionHeader")
        recent_col.addWidget(header)
        recent_col.addSpacing(8)

        self.recent_widget = QListWidget()
        self.recent_widget.setFixedWidth(300)
        self.recent_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recent_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.recent_widget.setCursor(Qt.PointingHandCursor)
        self.recent_widget.itemDoubleClicked.connect(self._open_recent)
        recent_col.addWidget(self.recent_widget)

        center.addLayout(recent_col)
        center.addStretch()
        root.addLayout(center)

        root.addStretch(1)
        self._populate_recent()

    def _populate_recent(self):

        self.recent_widget.clear()
        for entry in self.recent_list[:MAX_RECENT]:
            kind = "Folder" if entry["kind"] == "folder" else "Volume"
            name = os.path.basename(entry["path"]) or entry["path"]
            age = _age(entry.get("timestamp", ""))
            item = QListWidgetItem(f"{kind}  {name}    {age}\n      {entry['path']}")
            item.setData(Qt.UserRole, entry)
            self.recent_widget.addItem(item)

        if not self.recent_list:
            placeholder = QListWidgetItem("No recent files")
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemIsEnabled)
            self.recent_widget.addItem(placeholder)

        self._fit_recent_height()

    def _fit_recent_height(self):

        count = self.recent_widget.count()
        if count == 0:
            self.recent_widget.setFixedHeight(0)
            return
        total = sum(
            self.recent_widget.sizeHintForRow(i) for i in range(count)
        )
        frame = self.recent_widget.frameWidth() * 2
        padding = 8
        self.recent_widget.setFixedHeight(total + frame + padding)

    def refresh_recent(self, recent_list):

        self.recent_list = recent_list
        self._populate_recent()

    def _pick_folder(self):

        path = QFileDialog.getExistingDirectory(self, "Select Sparse OCT Folder")
        if path:
            self.open_folder.emit(path)

    def _pick_volume(self):

        path, _ = QFileDialog.getOpenFileName(self, "Select Volume", filter="NumPy (*.npz *.npy)")
        if path:
            self.open_volume.emit(path)

    def _open_recent(self, item):

        entry = item.data(Qt.UserRole)
        if not entry:
            return
        if entry["kind"] == "folder":
            self.open_folder.emit(entry["path"])
        else:
            self.open_volume.emit(entry["path"])
