import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

NUM_CLASSES = 6
INPAINT_BATCH = 16
DINO_BATCH = 16
COL_STRIDE = 20
DENSE_SIZE = 512
IMAGE_SIZE = 512
SLICE_SIZE = (512, 512)
NUM_SLICES = 25
PROB_SMOOTH_SIGMA = 1.5
DEPTH_SMOOTH_SIGMA = 2.5
MEDIAN_SMOOTH_KERNEL = 9

DINO_WEIGHTS = os.path.join(ROOT_DIR, "weights", "best_dino_segmenter_finetuned_vits.pth")
CHECKPOINT_PATH_RES = os.path.join(ROOT_DIR, "weights", "best_model_17k_4layers_32ch.pth")

CLASS_NAMES = [
    "ILM / RNFL",
    "GCL + IPL",
    "INL / OPL",
    "ONL / IS-OS",
    "RPE",
    "Choroid / BG",
]

CLASS_COLORS = np.array([
    [31, 119, 180, 180],
    [255, 127, 14, 180],
    [44, 160, 44, 180],
    [214, 39, 40, 180],
    [148, 103, 189, 180],
    [140, 86, 75, 180],
], dtype=np.uint8)
