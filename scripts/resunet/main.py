import os
import argparse
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from scripts.resunet.model import ResUNet
from scripts.resunet.dataset import OCTDataset
from scripts.resunet.flip_loss import LDRFLIPLoss
from scripts.resunet.training import train, validate
from scripts.resunet.visualize import visualize_results
from scripts.resunet.config import ALPHA, SCHEDULER_PATIENCE, WEIGHTS_PATH

def get_args():

    parser = argparse.ArgumentParser(description="OCT Volume Reconstruction")
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--data_root', type=str, default=os.environ.get("RESUNET_DATA_ROOT"))
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=6)

    return parser.parse_args()

def main():

    load_dotenv()

    args = get_args()
    torch.manual_seed(42)
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Config: {args.batch_size} BS | {args.lr} LR | {device}")

    train_data_root = os.path.join(args.data_root, "img_train")
    dataset = OCTDataset(train_data_root, max_samples=args.max_samples)
    train_idx, val_idx = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_idx, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_idx, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, persistent_workers=True, prefetch_factor=4)

    model = ResUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.L1Loss()
    criterion2 = LDRFLIPLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=SCHEDULER_PATIENCE, factor=0.5, min_lr=1e-6)

    best_loss = float('inf')

    for epoch in range(args.epochs):

        t_loss = train(model, train_loader, optimizer, criterion, criterion2, device, ALPHA)
        v_loss, v_ssim, v_ms_ssim, v_flip = validate(model, val_loader, criterion, criterion2, device, ALPHA)

        scheduler.step(v_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d} | LR: {current_lr:.6f} | Train: {t_loss:.4f} | Val: {v_loss:.4f} | SSIM: {v_ssim:.4f} | MS-SSIM: {v_ms_ssim:.4f} | FLIP: {v_flip:.4f}")

        if v_loss < best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), WEIGHTS_PATH)

        if (epoch + 1) % 10 == 0:
            visualize_results(model, val_idx, device)

if __name__ == "__main__":
    main()