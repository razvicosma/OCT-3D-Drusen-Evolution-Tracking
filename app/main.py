import os
import sys

import napari
import numpy as np
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox

from app.ui.landing import LandingPage
from app.ui.loading import LoadingPage
from app.ui.settings import SettingsDialog
from app.workers.pipeline import PipelineWorker
from app.utils.recent import load_settings, save_settings, add_recent
from scripts.segmentation.config import NUM_CLASSES, CLASS_NAMES, CLASS_COLORS
from app.config import (
    APP_TITLE, STYLE_PATH, MIN_WINDOW_SIZE, DEFAULT_WINDOW_SIZE,
    FOLDER_STEPS_FULL, FOLDER_STEPS_NO_RECON, FOLDER_STEPS_NO_SEG, VOLUME_STEPS,
)


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(*MIN_WINDOW_SIZE)
        self.settings = load_settings()
        self._worker = None
        self._build()

    def _build(self):

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.landing = LandingPage(self.settings.get("recent", []))
        self.loading = LoadingPage()

        self.stack.addWidget(self.landing)
        self.stack.addWidget(self.loading)

        self.landing.open_folder.connect(self._start_folder)
        self.landing.open_volume.connect(self._start_volume)
        self.loading.cancelled.connect(self._cancel)

        self._build_menu()

    def _build_menu(self):

        bar = self.menuBar()

        file_menu = bar.addMenu("File")

        act_folder = QAction("Open Folder…", self)
        act_folder.setShortcut("Ctrl+O")
        act_folder.triggered.connect(self.landing._pick_folder)
        file_menu.addAction(act_folder)

        act_volume = QAction("Open Volume…", self)
        act_volume.setShortcut("Ctrl+Shift+O")
        act_volume.triggered.connect(self.landing._pick_volume)
        file_menu.addAction(act_volume)

        file_menu.addSeparator()

        act_quit = QAction("Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(QApplication.quit)
        file_menu.addAction(act_quit)

        settings_menu = bar.addMenu("Settings")

        act_prefs = QAction("Preferences…", self)
        act_prefs.setShortcut("Ctrl+,")
        act_prefs.triggered.connect(self._open_settings)
        settings_menu.addAction(act_prefs)

    def _open_settings(self):

        dlg = SettingsDialog(self.settings, self)
        if dlg.exec():
            self.settings.update(dlg.result_settings())
            save_settings(self.settings)

    def _start_folder(self, path):

        add_recent(self.settings, path, "folder")
        self.landing.refresh_recent(self.settings.get("recent", []))

        do_recon = self.settings.get("reconstruction", True)
        do_seg = self.settings.get("segmentation", True)

        if do_recon and do_seg:
            steps = FOLDER_STEPS_FULL
        elif do_recon:
            steps = FOLDER_STEPS_NO_SEG
        else:
            steps = FOLDER_STEPS_NO_RECON

        self.loading.setup(steps, subtitle=os.path.basename(path))
        self.stack.setCurrentIndex(1)
        self._run_worker("folder", path)

    def _start_volume(self, path):

        add_recent(self.settings, path, "volume")
        self.landing.refresh_recent(self.settings.get("recent", []))

        self.loading.setup(VOLUME_STEPS, subtitle=os.path.basename(path))
        self.stack.setCurrentIndex(1)
        self._run_worker("volume", path)

    def _cancel(self):

        if self._worker and self._worker.isRunning():
            self._worker.terminate()
        self.loading.stop()
        self.stack.setCurrentIndex(0)

    def _run_worker(self, mode, path):

        self._worker = PipelineWorker(mode, path, self.settings, self)
        self._worker.step_started.connect(self.loading.mark_active)
        self._worker.step_progress.connect(self.loading.update_progress)
        self._worker.step_done.connect(self.loading.mark_done)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_all_done(self, result):

        self.loading.stop()
        self._open_napari(result)

    def _on_error(self, msg):

        self.loading.stop()
        self.stack.setCurrentIndex(0)
        QMessageBox.critical(self, "Pipeline Error", f"An error occurred:\n\n{msg}")

    def _open_napari(self, result):

        self.hide()

        viewer = napari.Viewer(title="OCT Volume")

        if result["sparse_display"] is not None:
            layer = viewer.add_image(
                result["sparse_display"],
                name="Sparse Input",
                colormap="gray",
                opacity=1.0,
                rendering="attenuated_mip",
            )
            layer.visible = False

        viewer.add_image(
            result["dense_volume"],
            name="Reconstructed Volume",
            colormap="gray",
            opacity=1.0,
            rendering="attenuated_mip",
        )

        if result["class_vols"] is not None:
            for c in range(NUM_CLASSES):
                rgba = CLASS_COLORS[c].astype(float) / 255.0
                color_dict = {0: np.array([0, 0, 0, 0]), c + 1: rgba}
                layer = viewer.add_labels(
                    result["class_vols"][c],
                    name=f"Seg {c}: {CLASS_NAMES[c]}",
                    opacity=0.6,
                    iso_gradient_mode="smooth",
                )
                layer.color = color_dict
                layer.visible = False

        viewer.dims.ndisplay = 3
        viewer.window._qt_window.showMaximized()
        viewer.window._qt_window.destroyed.connect(self._on_napari_closed)

    def _on_napari_closed(self):

        self.stack.setCurrentIndex(0)
        self.show()
        self.raise_()


def run():

    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    with open(STYLE_PATH) as f:
        app.setStyleSheet(f.read())
    window = MainWindow()
    window.resize(*DEFAULT_WINDOW_SIZE)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
