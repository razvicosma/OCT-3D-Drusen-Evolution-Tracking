import os

import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from scripts.dino.model import DINOv3Segmenter
from scripts.segmentation.loader import normalise_batch
from scripts.segmentation.config import NUM_CLASSES, DINO_WEIGHTS, DINO_BATCH, SLICE_SIZE, PROB_SMOOTH_SIGMA, DEPTH_SMOOTH_SIGMA


def gaussian_kernel1d(sigma, device):

    radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))

    return kernel / kernel.sum()


def smooth_probs(probs, sigma):

    if sigma <= 0:
        return probs

    C = probs.shape[1]
    k1d = gaussian_kernel1d(sigma, probs.device)
    k = k1d.numel()
    pad = k // 2

    weight_h = k1d.view(1, 1, k, 1).repeat(C, 1, 1, 1)
    weight_w = k1d.view(1, 1, 1, k).repeat(C, 1, 1, 1)

    probs = F.conv2d(F.pad(probs, (0, 0, pad, pad), mode="replicate"), weight_h, groups=C)
    probs = F.conv2d(F.pad(probs, (pad, pad, 0, 0), mode="replicate"), weight_w, groups=C)

    return probs


def smooth_probs_depth(probs_vol, sigma):

    if sigma <= 0:
        return probs_vol

    C = probs_vol.shape[1]
    k1d = gaussian_kernel1d(sigma, probs_vol.device).to(probs_vol.dtype)
    k = k1d.numel()
    pad = k // 2

    weight = k1d.view(1, 1, k, 1, 1).repeat(C, 1, 1, 1, 1)
    x = probs_vol.permute(1, 0, 2, 3).unsqueeze(0)
    x = F.pad(x, (0, 0, 0, 0, pad, pad), mode="replicate")
    x = F.conv3d(x, weight, groups=C)

    return x.squeeze(0).permute(1, 0, 2, 3)


def load_dino(device, weights_path=None):

    model = DINOv3Segmenter(num_classes=NUM_CLASSES)
    path = weights_path if weights_path is not None else DINO_WEIGHTS

    if os.path.exists(path):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"Loaded DINOv3 weights: {path}")
    else:
        print(f"Weights not found: {path}")

    model.to(device).eval()

    return model


def free_dino(model):

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def build_front_batch(volume, indices):

    slices_np = np.stack([cv2.resize(volume[i], SLICE_SIZE, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0 for i in indices], axis=0)

    return torch.from_numpy(np.stack([slices_np, slices_np, slices_np], axis=1))


def run_segmentation(volume, device, weights_path=None, progress_fn=None, batch_size=DINO_BATCH):

    model = load_dino(device, weights_path=weights_path)
    D, H, W = volume.shape
    masks = np.zeros((D, H, W), dtype=np.uint8)

    with torch.no_grad():
        probs_slices = []
        iterator = tqdm(range(0, D, batch_size), desc="Segmenting slices") if progress_fn is None else range(0, D, batch_size)

        for i in iterator:
            if progress_fn:
                progress_fn(min(i + batch_size, D), D)

            idxs = list(range(i, min(i + batch_size, D)))
            batch_t = build_front_batch(volume, idxs)
            batch_gpu = normalise_batch(batch_t, device)
            out = model(batch_gpu)
            probs = smooth_probs(F.softmax(out, dim=1), PROB_SMOOTH_SIGMA)
            probs_slices.append(probs)

            del batch_t, batch_gpu, out, probs

        probs_vol = torch.cat(probs_slices, dim=0)
        del probs_slices

        probs_vol = smooth_probs_depth(probs_vol, DEPTH_SMOOTH_SIGMA)
        preds = torch.argmax(probs_vol, dim=1).cpu().numpy()
        del probs_vol

        for i in range(D):
            masks[i] = cv2.resize(preds[i].astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

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
            probs = smooth_probs(F.softmax(out, dim=1), PROB_SMOOTH_SIGMA)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            store_side_predictions(preds, masks, xs, H, D)

            del batch_t, batch_gpu, out, probs

    free_dino(model)

    return masks
