import os
import numpy as np
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_CLASSES = 6
IMAGE_SIZE = 512
SCHEDULER_PATIENCE = 10

VAL_DICE_CEILING = 0.0741
CONTIG_PLATEAU_WINDOW = 5
CONTIG_PLATEAU_TOL = 0.05

AUG_HFLIP_P = 0.5
AUG_CROP_SCALE = (0.7, 1.0)
AUG_BRIGHTNESS_JITTER = 0.2
AUG_CONTRAST_JITTER = 0.2
AUG_NOISE_STD = 0.02


BACKBONE_WEIGHTS = os.path.join(ROOT_DIR, "weights", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights", "best_dino_segmenter_vits.pth")
FINETUNE_WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights", "best_dino_segmenter_finetuned_vits.pth")

UNLABELED_DIR = os.environ.get("DINO_UNLABELED_ROOT")
VIS_DIR = os.environ.get("DINO_VIS_ROOT")
DATA_DIR = os.environ.get("DINO_DATA_ROOT")

PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
CONFUSION_MATRIX_PLOT_PATH = os.path.join(PLOTS_DIR, "latest_confusion_matrix.png")
PREDICTIONS_PLOT_PATH = os.path.join(PLOTS_DIR, "latest_predictions.png")
