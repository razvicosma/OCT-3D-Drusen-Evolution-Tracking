import os
import gc
import argparse
from datetime import datetime
from dotenv import load_dotenv

import cv2
import torch
import numpy as np

from scripts.segmentation.loader import load_volume
from scripts.segmentation.visualize import launch_viewer
from scripts.segmentation.segmentation import run_segmentation
from scripts.segmentation.postprocess import build_class_volumes
from scripts.segmentation.interpolation import resunet_interpolate
from scripts.segmentation.config import CHECKPOINT_PATH_RES, DENSE_SIZE, COL_STRIDE, SLICE_SIZE

def build_sparse_display(sparse_volume):

    display = np.zeros((DENSE_SIZE, SLICE_SIZE[0], SLICE_SIZE[1]), dtype=np.uint8)

    for s in range(sparse_volume.shape[0]):
        pos = s * COL_STRIDE
        if pos < DENSE_SIZE:
            display[pos] = cv2.resize(sparse_volume[s], SLICE_SIZE, interpolation=cv2.INTER_AREA).astype(np.uint8)

    return display

def discover_volume_folders(data_root):

    folders = []

    for d in os.listdir(data_root):
        if "2025" in d or "2026" in d:
            continue
        try:
            date_obj = datetime.strptime(d, '%m.%d.%Y')
            folders.append((date_obj, d))
        except ValueError:
            pass

    folders.sort(key=lambda x: x[0])

    return folders

def process_volume(patient_dir, device):

    sparse_volume, _ = load_volume(patient_dir)
    sparse_display = build_sparse_display(sparse_volume)
    dense_volume = resunet_interpolate(sparse_volume, device, checkpoint=CHECKPOINT_PATH_RES)
    dense_masks = run_segmentation(dense_volume, device)
    del sparse_volume

    class_vols = build_class_volumes(dense_masks)
    del dense_masks

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return sparse_display, dense_volume, class_vols

def stack_class_volumes(all_class_vols):

    num_classes = len(all_class_vols[0])

    return [np.stack([cv[c] for cv in all_class_vols], axis=0) for c in range(num_classes)]

def main():

    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.environ.get("SEGMENTATION_DATA_ROOT"))
    args = parser.parse_args()

    data_root = args.data_root

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}  |  Data Root: {data_root}")

    folders = discover_volume_folders(data_root)
    folders = [folders[i] for i in [0, 1, 2, 3, 4]]

    sparse_displays = []
    dense_volumes = []
    all_class_vols = []

    for _, d in folders:
        patient_dir = os.path.join(data_root, d, "od")
        if not os.path.isdir(patient_dir):
            print(f"Directory {patient_dir} not found, skipping.")
            continue

        print(f"Processing {d}")

        sparse_display, dense_volume, class_vols = process_volume(patient_dir, device)

        sparse_displays.append(sparse_display)
        dense_volumes.append(dense_volume)
        all_class_vols.append(class_vols)

    if not sparse_displays:
        print("No valid volumes found.")
        return

    sparse_display_4d = np.stack(sparse_displays, axis=0)
    dense_volume_4d = np.stack(dense_volumes, axis=0)
    class_vols_4d = stack_class_volumes(all_class_vols)

    print(f"4D Sparse Display: {sparse_display_4d.shape}")
    print(f"4D Dense Volume: {dense_volume_4d.shape}")

    launch_viewer(sparse_display_4d, dense_volume_4d, class_vols_4d)

if __name__ == "__main__":
    main()
