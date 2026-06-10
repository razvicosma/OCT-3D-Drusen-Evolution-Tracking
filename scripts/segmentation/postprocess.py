import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from scripts.segmentation.config import NUM_CLASSES

def nan_fill(x):

    filled = x.clone()
    W = x.shape[1]

    for shift in range(1, W):
        mask = ~torch.isfinite(filled)
        filled[mask] = torch.roll(filled, -shift, dims=1)[mask]

    for shift in range(1, W):
        mask = ~torch.isfinite(filled)
        filled[mask] = torch.roll(filled, shift, dims=1)[mask]

    return filled

def median_filter_1d(x, kernel, pad):

    finite = torch.isfinite(x)

    if not finite.any():
        return x

    filled = nan_fill(x)
    padded = F.pad(filled, (pad, pad), mode="replicate")
    windows = padded.unfold(1, kernel, 1)
    smoothed = windows.median(dim=-1).values
    smoothed[~finite] = float("nan")

    return smoothed

def class_boundaries(binary, H, device):

    D, _, W = binary.shape
    has_any = binary.any(dim=1)

    top = torch.where(
        has_any,
        binary.to(torch.float32).argmax(dim=1).to(torch.float32),
        torch.full((D, W), float("nan"), device=device),
    )

    bot = torch.where(
        has_any,
        (H - 1 - binary.flip(1).to(torch.float32).argmax(dim=1)).to(torch.float32),
        torch.full((D, W), float("nan"), device=device),
    )

    return top, bot

def fill_between_boundaries(top_s, bot_s, H, device):

    row_idx = torch.arange(H, device=device).reshape(1, H, 1).to(torch.float32)
    t0 = top_s.unsqueeze(1).nan_to_num(nan=H)
    t1 = bot_s.unsqueeze(1).nan_to_num(nan=-1)

    return (row_idx >= t0) & (row_idx <= t1)

def smooth_mask_edges(masks, kernel=9, device=torch.device("cpu")):

    D, H, W = masks.shape
    pad = kernel // 2
    out_t = torch.zeros((D, H, W), dtype=torch.uint8, device=device)
    masks_t = torch.from_numpy(masks).to(device)

    for c in tqdm(range(NUM_CLASSES), desc="Smoothing edges"):
        binary = (masks_t == c)

        top, bot = class_boundaries(binary, H, device)
        top_s = median_filter_1d(top, kernel, pad)
        bot_s = median_filter_1d(bot, kernel, pad)
        fill = fill_between_boundaries(top_s, bot_s, H, device)

        out_t[fill] = c

        del binary, top, bot, top_s, bot_s, fill

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()

    out = out_t.cpu().numpy()

    del masks_t, out_t

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return out

def build_class_volumes(dense_masks):

    classes = np.arange(NUM_CLASSES, dtype=np.uint8)[:, None, None, None]

    return list((dense_masks == classes) * (classes + 1))
