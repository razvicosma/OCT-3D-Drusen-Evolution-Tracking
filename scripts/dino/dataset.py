import os
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from scripts.dino.config import (
    IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE,
    AUG_HFLIP_P, AUG_CROP_SCALE, AUG_BRIGHTNESS_JITTER, AUG_CONTRAST_JITTER, AUG_NOISE_STD
)

class OCTDataset(Dataset):

    def __init__(self, img_dir, msk_dir):

        img_dir = str(img_dir)
        msk_dir = str(msk_dir)

        valid_exts = ('.jpeg', '.jpg', '.png', '.tif', '.tiff')
        img_stems = {}

        for f in os.listdir(img_dir):
            if f.endswith(valid_exts):
                img_stems[os.path.splitext(f)[0]] = os.path.join(img_dir, f)

        msk_stems = {}

        for f in os.listdir(msk_dir):
            if f.endswith(valid_exts):
                msk_stems[os.path.splitext(f)[0]] = os.path.join(msk_dir, f)

        common = sorted(img_stems.keys() & msk_stems.keys())
        self.pairs = [(img_stems[s], msk_stems[s]) for s in common]

        if len(self.pairs) == 0:
            raise RuntimeError(f"No matching pairs found between {img_dir} and {msk_dir}")

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):

        img_path, msk_path = self.pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        mask = mask.astype(np.int64)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        return image, mask

class AugmentedOCTDataset(Dataset):

    def __init__(self, base_dataset, hflip_p=AUG_HFLIP_P, crop_scale=AUG_CROP_SCALE,
                 brightness_jitter=AUG_BRIGHTNESS_JITTER, contrast_jitter=AUG_CONTRAST_JITTER, noise_std=AUG_NOISE_STD):

        self.ds = base_dataset
        self.hflip_p = hflip_p
        self.crop_scale = crop_scale
        self.brightness_jitter = brightness_jitter
        self.contrast_jitter = contrast_jitter
        self.noise_std = noise_std

        self.mean_t = torch.from_numpy(IMAGENET_MEAN).to(torch.float32).reshape(3, 1, 1)
        self.std_t = torch.from_numpy(IMAGENET_STD).to(torch.float32).reshape(3, 1, 1)

    def __len__(self):

        return len(self.ds)

    def __getitem__(self, idx):

        image, mask = self.ds[idx]
        H, W = image.shape[1], image.shape[2]

        img_np = image.permute(1, 2, 0).numpy()
        msk_np = mask.numpy()

        if random.random() < self.hflip_p:
            img_np = cv2.flip(img_np, 1)
            msk_np = cv2.flip(msk_np, 1)

        scale = random.uniform(*self.crop_scale)
        crop_h = int(H * scale)
        crop_w = int(W * scale)
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        img_crop = img_np[top:top+crop_h, left:left+crop_w]
        msk_crop = msk_np[top:top+crop_h, left:left+crop_w]

        img_np = cv2.resize(img_crop, (W, H), interpolation=cv2.INTER_LINEAR)
        msk_np = cv2.resize(msk_crop, (W, H), interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(img_np).permute(2, 0, 1)
        mask = torch.from_numpy(msk_np)

        image = image * self.std_t + self.mean_t

        if self.brightness_jitter > 0:
            delta = random.uniform(-self.brightness_jitter, self.brightness_jitter)
            image = image + delta

        if self.contrast_jitter > 0:
            factor = random.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
            mean = image.mean(dim=(1, 2), keepdim=True)
            image = (image - mean) * factor + mean

        if self.noise_std > 0:
            image = image + torch.randn_like(image) * self.noise_std

        image = torch.clamp(image, 0.0, 1.0)
        image = (image - self.mean_t) / self.std_t

        return image, mask

class UnsupervisedOCTDataset(Dataset):

    def __init__(self, root_dir):

        self.image_paths = []
        valid_exts = ('.jpeg', '.jpg', '.png', '.tif', '.tiff')

        for dirpath, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.lower().endswith(valid_exts):
                    self.image_paths.append(os.path.join(dirpath, f))

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

        mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.int64)

        image = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        mask  = torch.from_numpy(mask)

        return image, mask
