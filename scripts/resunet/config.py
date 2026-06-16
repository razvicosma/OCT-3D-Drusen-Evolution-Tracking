import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_SIZE = (512, 512)
ALPHA = 0.7
SCHEDULER_PATIENCE = 10
PERIOD_RANGE = (6, 20)
PIXELS_PER_DEGREE = (0.7 * 3840 / 0.7) * np.pi / 180

WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights", "best_model_17k.pth")
DINO_REFERENCE_IMAGE = os.path.join(ROOT_DIR, "015_Drusen.tif")
DINO_FEAT_PATH = os.path.join(ROOT_DIR, "weights", "dino_ref_feat.pt")

PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
RECONSTRUCTION_PLOT_PATH = os.path.join(PLOTS_DIR, "reconstruction_results.png")
HEATMAPS_PLOT_PATH = os.path.join(PLOTS_DIR, "heatmaps_results.png")
