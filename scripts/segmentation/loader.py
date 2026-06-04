import os

import cv2
import torch
import numpy as np

from scripts.segmentation.config import SLICE_SIZE, NUM_SLICES, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE

MEAN = torch.from_numpy(IMAGENET_MEAN).to(torch.float32).reshape(1, 3, 1, 1)
STD = torch.from_numpy(IMAGENET_STD).to(torch.float32).reshape(1, 3, 1, 1)

def load_volume(patient_dir):

    valid_exts = ('.jpeg', '.jpg', '.png', '.tif', '.tiff')
    files = []

    for f in os.listdir(patient_dir):
        if f.endswith(valid_exts):
            files.append(os.path.join(patient_dir, f))

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No images found in {patient_dir}")

    slices = []

    for fp in files:
        img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, SLICE_SIZE, interpolation=cv2.INTER_AREA)
        slices.append(img)

    while len(slices) < NUM_SLICES:
        slices.append(slices[-1].copy())

    volume = np.stack(slices[:NUM_SLICES], axis=0).astype(np.float32)

    return volume, files[:NUM_SLICES]

def preprocess_for_dino(gray_slice):

    resized = cv2.resize(gray_slice, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)
    t = torch.from_numpy(arr).unsqueeze(0)

    return t

def normalise_batch(batch, device):

    return (batch.to(device) - MEAN.to(device)) / STD.to(device)
