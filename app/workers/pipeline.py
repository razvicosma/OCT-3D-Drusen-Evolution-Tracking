import os
import cv2
import numpy as np
import torch

from PySide6.QtCore import QThread, Signal

from scripts.segmentation.loader import load_volume
from scripts.segmentation.interpolation import resunet_interpolate
from scripts.segmentation.segmentation import run_segmentation
from scripts.segmentation.postprocess import build_class_volumes
from scripts.segmentation.config import (
    DENSE_SIZE, COL_STRIDE, SLICE_SIZE,
    DINO_WEIGHTS, DINO_WEIGHTS_FINE,
    INPAINT_BATCH,
)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PipelineWorker(QThread):
    step_started = Signal(str, int)
    step_progress = Signal(int, int)
    step_done = Signal(str)
    all_done = Signal(object)
    error = Signal(str)

    def __init__(self, mode, path, settings, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.path = path
        self.settings = settings

    def run(self):
        try:
            if self.mode == "volume":
                self._run_volume()
            else:
                self._run_folder()
        except Exception as e:
            import traceback
            self.error.emit(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")

    def _start(self, name, total=0):
        self.step_started.emit(name, total)

    def _progress(self, current, total):
        self.step_progress.emit(current, total)

    def _done(self, name):
        self.step_done.emit(name)

    def _weights(self):
        return DINO_WEIGHTS_FINE if self.settings.get("model") == "dinov3fine" else DINO_WEIGHTS

    def _build_sparse_display(self, sparse_volume):
        display = np.zeros((DENSE_SIZE, SLICE_SIZE[0], SLICE_SIZE[1]), dtype=np.uint8)
        for s in range(sparse_volume.shape[0]):
            pos = s * COL_STRIDE
            if pos < DENSE_SIZE:
                display[pos] = cv2.resize(sparse_volume[s], SLICE_SIZE, interpolation=cv2.INTER_AREA).astype(np.uint8)
        return display

    def _run_volume(self):
        self._start("Loading volume")
        if self.path.endswith(".npz"):
            data = np.load(self.path)
            dense_volume = data["dense_volume"]
            class_vols = data["class_vols"] if "class_vols" in data else None
        else:
            dense_volume = np.load(self.path)
            class_vols = None
        self._done("Loading volume")

        self.all_done.emit({
            "sparse_display": None,
            "dense_volume": dense_volume,
            "class_vols": class_vols,
            "save_path": self.path,
        })

    def _run_folder(self):
        device = _device()
        do_recon = self.settings.get("reconstruction", True)
        do_seg = self.settings.get("segmentation", True)

        self._start("Loading scans")
        sparse_volume, _ = load_volume(self.path)
        self._done("Loading scans")
        sparse_display = self._build_sparse_display(sparse_volume)

        if do_recon:
            total = len(list(range(0, sparse_volume.shape[2], INPAINT_BATCH)))
            self._start("Reconstructing volume", total)
            dense_volume = resunet_interpolate(sparse_volume, device, progress_fn=self._progress)
            self._done("Reconstructing volume")
        else:
            dense_volume = sparse_volume

        class_vols = None
        if do_seg:
            D = dense_volume.shape[0]
            self._start("Segmenting layers", D)
            dense_masks = run_segmentation(dense_volume, device, weights_path=self._weights(), progress_fn=self._progress)
            self._done("Segmenting layers")

            class_vols = build_class_volumes(dense_masks)

        self._start("Saving volume")
        save_path = os.path.join(self.path, "dense_volume.npz")
        if class_vols is not None:
            np.savez_compressed(save_path, dense_volume=dense_volume, class_vols=class_vols)
        else:
            np.savez_compressed(save_path, dense_volume=dense_volume)
        self._done("Saving volume")

        self.all_done.emit({
            "sparse_display": sparse_display,
            "dense_volume": dense_volume,
            "class_vols": class_vols,
            "save_path": save_path,
        })
