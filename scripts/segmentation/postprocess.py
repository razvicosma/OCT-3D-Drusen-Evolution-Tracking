import torch
import numpy as np
import torch.nn.functional as F

from scripts.segmentation.config import NUM_CLASSES

def median_smooth_w(masks, kernel=3, device=torch.device("cpu")):

    D, H, W = masks.shape
    pad = kernel // 2

    x = torch.from_numpy(masks).to(device).float().reshape(D * H, W)
    padded = F.pad(x.unsqueeze(0), (pad, pad), mode="replicate").squeeze(0)
    windows = padded.unfold(1, kernel, 1)
    smoothed = windows.median(dim=-1).values

    result = smoothed.reshape(D, H, W).round().to(torch.uint8).cpu().numpy()

    del x, padded, windows, smoothed

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result

def build_class_volumes(dense_masks):

    classes = np.arange(NUM_CLASSES, dtype=np.uint8)[:, None, None, None]

    return list((dense_masks == classes) * (classes + 1))
