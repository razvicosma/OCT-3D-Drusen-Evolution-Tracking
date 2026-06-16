from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, QRect, QPointF, Signal
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QPolygonF

PENDING = 0
ACTIVE  = 1
DONE    = 2


class StepIcon(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 22)
        self._state = PENDING
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._rotate)

    def set_state(self, state):
        self._state = state
        if state == ACTIVE:
            self._angle = 0
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _rotate(self):
        self._angle = (self._angle + 5) % 360
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        m = 3
        rect = QRect(m, m, self.width() - 2 * m, self.height() - 2 * m)

        if self._state == PENDING:
            p.setPen(QPen(QColor("#374151"), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(rect)

        elif self._state == ACTIVE:
            p.setPen(QPen(QColor("#1f2937"), 2.0))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(rect)

            pen = QPen(QColor("#3b82f6"), 2.5)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            start_angle = (90 - self._angle) * 16
            span_angle  = -270 * 16
            p.drawArc(rect, start_angle, span_angle)

        elif self._state == DONE:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor("#22c55e")))
            p.drawEllipse(rect)

            pen = QPen(QColor("white"), 2.0)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            cx = self.width() / 2.0
            cy = self.height() / 2.0
            r  = (self.width() / 2.0 - m) * 0.58
            pts = QPolygonF([
                QPointF(cx - r * 0.70, cy + r * 0.05),
                QPointF(cx - r * 0.10, cy + r * 0.65),
                QPointF(cx + r * 0.72, cy - r * 0.52),
            ])
            p.drawPolyline(pts)


class StepItem(QWidget):

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.state = PENDING

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(14)

        self.icon = StepIcon()
        self.label = QLabel(name)

        row.addWidget(self.icon)
        row.addWidget(self.label)
        row.addStretch()
        layout.addLayout(row)

        self.bar = QProgressBar()
        self.bar.setMaximumHeight(5)
        self.bar.setTextVisible(False)
        self.bar.setVisible(False)
        layout.addWidget(self.bar)

        self._apply_label()

    def _apply_label(self):
        if self.state == PENDING:
            self.label.setStyleSheet("color: #4b5563;")
        elif self.state == ACTIVE:
            self.label.setStyleSheet("color: #f3f4f6; font-weight: 600;")
        elif self.state == DONE:
            self.label.setStyleSheet("color: #6b7280;")

    def set_state(self, state, total=0):
        self.state = state
        self.icon.set_state(state)
        self._apply_label()
        if state == ACTIVE:
            if total > 0:
                self.bar.setMaximum(total)
                self.bar.setValue(0)
            else:
                self.bar.setMaximum(0)
            self.bar.setVisible(True)
        else:
            self.bar.setVisible(False)

    def set_progress(self, current, total):
        if self.state == ACTIVE and total > 0:
            self.bar.setMaximum(total)
            self.bar.setValue(current)


class LoadingPage(QWidget):
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: dict[str, StepItem] = {}
        self._active = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 32)
        root.setSpacing(0)

        root.addStretch(1)

        self.title = QLabel("Processing…")
        self.title.setObjectName("loadingTitle")
        self.title.setAlignment(Qt.AlignCenter)
        root.addWidget(self.title)
        root.addSpacing(6)

        self.subtitle = QLabel("")
        self.subtitle.setObjectName("loadingSubtitle")
        self.subtitle.setAlignment(Qt.AlignCenter)
        root.addWidget(self.subtitle)
        root.addSpacing(36)

        center = QHBoxLayout()
        center.addStretch()

        self.steps_layout = QVBoxLayout()
        self.steps_layout.setSpacing(10)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        center.addLayout(self.steps_layout)
        center.addStretch()
        root.addLayout(center)

        root.addStretch(1)

        cancel_row = QHBoxLayout()
        cancel_row.addStretch()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("cancelButton")
        self.cancel_btn.clicked.connect(self.cancelled)
        cancel_row.addWidget(self.cancel_btn)
        root.addLayout(cancel_row)

    def setup(self, step_names, subtitle=""):
        while self.steps_layout.count():
            w = self.steps_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._items.clear()
        self._active = None
        self.subtitle.setText(subtitle)
        for name in step_names:
            item = StepItem(name)
            self._items[name] = item
            self.steps_layout.addWidget(item)

    def mark_active(self, name, total=0):
        self._active = name
        if name in self._items:
            self._items[name].set_state(ACTIVE, total)

    def mark_done(self, name):
        if name in self._items:
            self._items[name].set_state(DONE)

    def update_progress(self, current, total):
        if self._active and self._active in self._items:
            self._items[self._active].set_progress(current, total)

    def stop(self):
        for item in self._items.values():
            item.icon._timer.stop()
