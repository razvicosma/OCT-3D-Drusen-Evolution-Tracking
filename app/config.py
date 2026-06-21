import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

APP_TITLE = "OCT Drusen Tracker"
MIN_WINDOW_SIZE = (680, 520)
DEFAULT_WINDOW_SIZE = (720, 580)

MAX_RECENT = 3
SETTINGS_PATH = os.path.join(ROOT_DIR, "settings.json")
STYLE_PATH = os.path.join(ROOT_DIR, "ui", "style.qss")

PAGE_MARGINS = (40, 40, 40, 32)

STEP_PENDING = 0
STEP_ACTIVE = 1
STEP_DONE = 2

MODEL_BASE = "dinov3"
MODEL_FINE = "dinov3fine"

FOLDER_STEPS_FULL     = ["Loading scans", "Reconstructing volume", "Segmenting layers", "Saving volume"]
FOLDER_STEPS_NO_RECON = ["Loading scans", "Segmenting layers", "Saving volume"]
FOLDER_STEPS_NO_SEG   = ["Loading scans", "Reconstructing volume", "Saving volume"]
VOLUME_STEPS          = ["Loading volume"]
