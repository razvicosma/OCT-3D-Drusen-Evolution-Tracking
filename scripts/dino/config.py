import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
NUM_CLASSES = 6
IMAGE_SIZE = 512
SCHEDULER_PATIENCE = 10

AUG_HFLIP_P = 0.5
AUG_CROP_SCALE = (0.7, 1.0)
AUG_BRIGHTNESS_JITTER = 0.2
AUG_CONTRAST_JITTER = 0.2
AUG_NOISE_STD = 0.02


BACKBONE_WEIGHTS = os.path.join(ROOT_DIR, "weights", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights", "best_dino_segmenter.pth")

PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(PLOTS_DIR, "latest_confusion_matrix.png")
PREDICTIONS_PLOT_PATH = os.path.join(PLOTS_DIR, "latest_predictions.png")
