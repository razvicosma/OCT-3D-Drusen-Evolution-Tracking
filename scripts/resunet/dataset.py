import os
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from scripts.resunet.config import IMAGE_SIZE, PERIOD_RANGE

def create_strips(image, n_missing, include_offset=True):

    period = n_missing + 1
    offset = random.randint(0, period - 1) if include_offset else 0
    
    cols = np.arange(image.shape[1])
    
    is_missing = (cols > offset) & ((cols - offset) % period != 0)
    
    stripped = image.copy()
    stripped[:, is_missing] = 0.0
    
    mask = np.ones_like(image)
    mask[:, is_missing] = 0.0
    
    return stripped, mask

class OCTDataset(Dataset):

    def __init__(self, root_dir, img_size=IMAGE_SIZE, period_range=PERIOD_RANGE, max_samples=None):

        self.img_size = img_size
        self.period_range = period_range
        self.paths = []
        
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Root_dir does not exist: {root_dir!r}")

        for f in os.listdir(root_dir):
            if f.endswith(('.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                self.paths.append(os.path.join(root_dir, f))

        if max_samples is not None:
            self.paths = self.paths[:max_samples]

    def __len__(self):

        return len(self.paths)
    
    def __getitem__(self, idx):

        img = cv2.imread(self.paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0

        n_missing = random.randint(*self.period_range)

        stripped, mask = create_strips(img, n_missing)

        img_tensor = torch.from_numpy(img).unsqueeze(0)
        stripped_tensor = torch.from_numpy(stripped).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return stripped_tensor, img_tensor, mask_tensor