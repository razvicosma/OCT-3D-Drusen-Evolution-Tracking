import os

import cv2
import torch
import numpy as np
from tqdm import tqdm

from scripts.resunet.model import ResUNet
from scripts.segmentation.config import CHECKPOINT_PATH_RES, INPAINT_BATCH, COL_STRIDE, SLICE_SIZE, IMAGE_SIZE


def build_canvas(sparse_plane, num_slices, col_stride=COL_STRIDE):

    canvas = np.zeros(SLICE_SIZE, dtype=np.float32)
    mask = np.zeros(SLICE_SIZE, dtype=np.float32)
    plane_r = cv2.resize(sparse_plane / 255.0, (num_slices, SLICE_SIZE[0]), interpolation=cv2.INTER_AREA)
    for s in range(num_slices):
        c = s * col_stride
        if c < SLICE_SIZE[1]:
            canvas[:, c] = plane_r[:, s]
            mask[:, c] = 1.0

    return canvas, mask

def load_resunet(checkpoint, device):

    model = ResUNet().to(device)

    if os.path.exists(checkpoint):
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        print(f"Loaded ResUNet: {checkpoint}")
    else:
        print(f"ResUNet checkpoint not found: {checkpoint}")

    model.eval()

    return model

def resunet_interpolate(sparse_volume, device, checkpoint=CHECKPOINT_PATH_RES, batch_size=INPAINT_BATCH):

    model = load_resunet(checkpoint, device)
    num_slices, height, width = sparse_volume.shape
    inpaint_vol = np.zeros((width, height, IMAGE_SIZE), dtype=np.uint8)

    for i in tqdm(range(0, width, batch_size), desc="Interpolating"):
        curr = min(batch_size, width - i)
        canvases, masks_inp = [], []

        for j in range(curr):
            col_idx = i + j
            sparse_plane = sparse_volume[:, :, col_idx].T
            canvas, mask = build_canvas(sparse_plane, num_slices)
            canvases.append(canvas)
            masks_inp.append(mask)

        in_c = torch.from_numpy(np.stack(canvases)).float().to(device).unsqueeze(1)
        in_m = torch.from_numpy(np.stack(masks_inp)).float().to(device).unsqueeze(1)
        inp = torch.cat([in_c, in_m], dim=1)

        with torch.no_grad():
            preds = model(inp).squeeze(1).cpu().numpy()

        for j in range(curr):
            inpaint_vol[i + j] = (np.clip(preds[j], 0, 1) * 255.0).astype(np.uint8)

        del in_c, in_m, inp, preds

    result = inpaint_vol.transpose(2, 1, 0)

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result
