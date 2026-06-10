import os

import cv2
import torch
import numpy as np
from tqdm import tqdm

from scripts.dino.model import DINOv3Segmenter
from scripts.segmentation.loader import preprocess_for_dino, normalise_batch
from scripts.segmentation.config import NUM_CLASSES, DINO_WEIGHTS, DINO_BATCH, SLICE_SIZE


def load_dino(device):

    model = DINOv3Segmenter(num_classes=NUM_CLASSES)

    if os.path.exists(DINO_WEIGHTS):
        model.load_state_dict(torch.load(DINO_WEIGHTS, map_location=device, weights_only=True))
        print(f"Loaded DINOv3 weights: {DINO_WEIGHTS}")
    else:
        print(f"Weights not found: {DINO_WEIGHTS}")

    model.to(device).eval()

    return model


def free_dino(model):

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_segmentation(volume, device):

    model = load_dino(device)
    D, H, W = volume.shape
    masks = np.zeros((D, H, W), dtype=np.uint8)

    with torch.no_grad():
        for i in tqdm(range(D), desc="Segmenting slices"):
            t_raw = preprocess_for_dino(volume[i])
            t = normalise_batch(t_raw, device)
            out = model(t)
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
            masks[i] = cv2.resize(pred.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            del t_raw, t, out, pred

    free_dino(model)

    return masks


def build_side_batch(volume, xs):

    slices_np = np.stack([cv2.resize(volume[:, :, x].T, SLICE_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0 for x in xs], axis=0)

    return torch.from_numpy(np.stack([slices_np, slices_np, slices_np], axis=1))


def store_side_predictions(preds, masks, xs, H, D):

    for k, x in enumerate(xs):
        pred_resized = cv2.resize(preds[k].astype(np.float32), (D, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        masks[:, :, x] = pred_resized.T


def run_segmentation_side_view(volume, device, batch_size=DINO_BATCH):

    model = load_dino(device)
    D, H, W = volume.shape
    masks = np.zeros((D, H, W), dtype=np.uint8)

    with torch.no_grad():
        for i in tqdm(range(0, W, batch_size), desc="Segmenting side slices"):
            xs = list(range(i, min(i + batch_size, W)))
            batch_t = build_side_batch(volume, xs)
            batch_gpu = normalise_batch(batch_t, device)
            out = model(batch_gpu)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            store_side_predictions(preds, masks, xs, H, D)

    del batch_gpu, out, preds
    free_dino(model)

    return masks
