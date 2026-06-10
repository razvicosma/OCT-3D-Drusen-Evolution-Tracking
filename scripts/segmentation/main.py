import os
import argparse
from dotenv import load_dotenv

import cv2
import torch
import numpy as np

from scripts.segmentation.loader import load_volume
from scripts.segmentation.visualize import launch_viewer
from scripts.segmentation.interpolation import resunet_interpolate
from scripts.segmentation.segmentation import run_segmentation_side_view
from scripts.segmentation.postprocess import smooth_mask_edges, build_class_volumes
from scripts.segmentation.config import CHECKPOINT_PATH_RES, DENSE_SIZE, COL_STRIDE, SLICE_SIZE, SMOOTHING_KERNEL

def build_sparse_display(sparse_volume):

    display = np.zeros((DENSE_SIZE, SLICE_SIZE[0], SLICE_SIZE[1]), dtype=np.uint8)

    for s in range(sparse_volume.shape[0]):
        pos = s * COL_STRIDE
        if pos < DENSE_SIZE:
            display[pos] = cv2.resize(sparse_volume[s], SLICE_SIZE, interpolation=cv2.INTER_AREA).astype(np.uint8)

    return display

def main():

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.environ.get("SEGMENTATION_DATA_ROOT"))
    args = parser.parse_args()

    patient_dir = args.data_root

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}  |  Patient: {patient_dir}")

    sparse_volume, _ = load_volume(patient_dir)
    print(f"Sparse volume: {sparse_volume.shape}")

    sparse_display = build_sparse_display(sparse_volume)

    dense_volume = resunet_interpolate(sparse_volume, device, checkpoint=CHECKPOINT_PATH_RES)
    print(f"Dense volume: {dense_volume.shape}")

    dense_masks = run_segmentation_side_view(dense_volume, device)
    print(f"Dense masks: {dense_masks.shape}")

    dense_masks = smooth_mask_edges(dense_masks, kernel=SMOOTHING_KERNEL, device=device)

    class_vols = build_class_volumes(dense_masks)

    launch_viewer(sparse_display, dense_volume, class_vols)

if __name__ == "__main__":
    main()
