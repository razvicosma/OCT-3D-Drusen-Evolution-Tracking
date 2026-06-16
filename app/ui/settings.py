from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QCheckBox,
    QRadioButton, QButtonGroup, QPushButton, QGroupBox
)
from PySide6.QtCore import Qt


class SettingsDialog(QDialog):

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(380)
        self._settings = dict(settings)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        pipe_box = QGroupBox("Pipeline  (folder mode)")
        pipe_layout = QVBoxLayout(pipe_box)
        pipe_layout.setSpacing(12)

        self.chk_recon = QCheckBox("Volume Reconstruction  (ResUNet interpolation)")
        self.chk_recon.setChecked(self._settings.get("reconstruction", True))

        self.chk_seg = QCheckBox("Layer Segmentation  (DINOv3)")
        self.chk_seg.setChecked(self._settings.get("segmentation", True))

        self.chk_recon.toggled.connect(self._enforce_at_least_one)
        self.chk_seg.toggled.connect(self._enforce_at_least_one)

        pipe_layout.addWidget(self.chk_recon)
        pipe_layout.addWidget(self.chk_seg)
        layout.addWidget(pipe_box)

        model_box = QGroupBox("Segmentation Model")
        model_layout = QVBoxLayout(model_box)
        model_layout.setSpacing(10)

        self.rb_base = QRadioButton("DINOv3  (base model)")
        self.rb_fine = QRadioButton("DINOv3fine  (domain-adapted)")

        group = QButtonGroup(self)
        group.addButton(self.rb_base)
        group.addButton(self.rb_fine)

        if self._settings.get("model") == "dinov3fine":
            self.rb_fine.setChecked(True)
        else:
            self.rb_base.setChecked(True)

        self.chk_seg.toggled.connect(self._toggle_model_enabled)
        self._toggle_model_enabled(self.chk_seg.isChecked())

        model_layout.addWidget(self.rb_base)
        model_layout.addWidget(self.rb_fine)
        layout.addWidget(model_box)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._accept)
        ok_btn.setDefault(True)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def _enforce_at_least_one(self):
        if not self.chk_recon.isChecked() and not self.chk_seg.isChecked():
            self.sender().blockSignals(True)
            self.sender().setChecked(True)
            self.sender().blockSignals(False)

    def _toggle_model_enabled(self, enabled):
        self.rb_base.setEnabled(enabled)
        self.rb_fine.setEnabled(enabled)

    def _accept(self):
        self._settings["reconstruction"] = self.chk_recon.isChecked()
        self._settings["segmentation"] = self.chk_seg.isChecked()
        self._settings["model"] = "dinov3fine" if self.rb_fine.isChecked() else "dinov3"
        self.accept()

    def result_settings(self):
        return self._settings
