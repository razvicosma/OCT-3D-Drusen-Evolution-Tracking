import os
import random
import argparse
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader

from scripts.dino.loss import CombinedLoss
from scripts.dino.model import DINOv3Segmenter
from scripts.dino.visualize import visualize_predictions
from scripts.dino.dataset import OCTDataset, AugmentedOCTDataset, UnsupervisedOCTDataset
from scripts.dino.training import build_optimizer, compute_class_weights, train, validate, train_unsupervised, validate_unsupervised
from scripts.dino.config import (
    NUM_CLASSES, SCHEDULER_PATIENCE, WEIGHTS_PATH,
    FINETUNE_WEIGHTS_PATH, UNLABELED_DIR, VIS_DIR, DATA_DIR
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "finetune"], default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--unfreeze_blocks", type=int, default=5)

    return parser.parse_args()

def main():
    
    load_dotenv()

    args = get_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = DINOv3Segmenter(num_classes=NUM_CLASSES).to(device)
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    if args.mode == "train":

        if os.path.exists(WEIGHTS_PATH):
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

        train_img_dir = os.path.join(DATA_DIR, "train_img")
        train_msk_dir = os.path.join(DATA_DIR, "train_msk")
        valid_img_dir = os.path.join(DATA_DIR, "valid_img")
        valid_msk_dir = os.path.join(DATA_DIR, "valid_msk")
        test_img_dir = os.path.join(DATA_DIR, "test_img")
        test_msk_dir = os.path.join(DATA_DIR, "test_msk")

        raw_train = OCTDataset(train_img_dir, train_msk_dir)
        val_dataset = OCTDataset(valid_img_dir, valid_msk_dir)
        test_dataset = OCTDataset(test_img_dir, test_msk_dir)

        train_dataset = AugmentedOCTDataset(raw_train)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)

        class_weights = compute_class_weights(raw_train, args.batch_size, args.num_workers, device)
        criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5, class_weights=class_weights)
        optimizer = build_optimizer(model, args)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=0.5, min_lr=1e-6)

        best_loss = float('inf')

        for epoch in range(args.epochs):
            avg_train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
            avg_val_loss, avg_val_ce, avg_val_dice = validate(model, val_loader, criterion, device)

            lr_head = optimizer.param_groups[0]['lr']
            lr_info = f"LR head: {lr_head:.2e}"
        
            if len(optimizer.param_groups) > 1:
                lr_info += f" | LR backbone: {optimizer.param_groups[1]['lr']:.2e}"

            print(f"Epoch {epoch+1:3d} | {lr_info} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | CE: {avg_val_ce:.4f} | Dice: {avg_val_dice:.4f}")

            scheduler.step(avg_val_loss)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), WEIGHTS_PATH)

            if (epoch + 1) % 10 == 0:
                visualize_predictions(model, val_dataset, device, args.batch_size, num_workers=args.num_workers)

    elif args.mode == "finetune":
        
        teacher_model = DINOv3Segmenter(num_classes=NUM_CLASSES).to(device)
        teacher_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        teacher_model.eval()

        for param in teacher_model.parameters():
            param.requires_grad = False
            
        if os.path.exists(FINETUNE_WEIGHTS_PATH):
            model.load_state_dict(torch.load(FINETUNE_WEIGHTS_PATH, map_location=device))
        elif os.path.exists(WEIGHTS_PATH):
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))

        full_vis = UnsupervisedOCTDataset(VIS_DIR)
        vis_indices = []
        
        for prefix in ('CNV', 'DME', 'NORMAL'):
            matches = [i for i, p in enumerate(full_vis.image_paths) if os.path.basename(p).startswith(prefix)]
            vis_indices.extend(random.sample(matches, min(2, len(matches))))
        cnv_vis_ds = torch.utils.data.Subset(full_vis, vis_indices)

        raw_train = UnsupervisedOCTDataset(UNLABELED_DIR)
        aug_train = AugmentedOCTDataset(raw_train)
        
        train_loader = DataLoader(aug_train, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

        train_img_sup = os.path.join(DATA_DIR, "train_img")
        train_msk_sup = os.path.join(DATA_DIR, "train_msk")
        valid_img_sup = os.path.join(DATA_DIR, "valid_img")
        valid_msk_sup = os.path.join(DATA_DIR, "valid_msk")
        
        val_train_dataset = OCTDataset(train_img_sup, train_msk_sup)
        val_valid_dataset = OCTDataset(valid_img_sup, valid_msk_sup)
        val_dataset = torch.utils.data.ConcatDataset([val_train_dataset, val_valid_dataset])
        
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)

        criterion = CombinedLoss(weight_ce=0.5, weight_dice=0.5, weight_sobel=1.0, weight_contiguity=1.0)
        
        optimizer = build_optimizer(model, args)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=SCHEDULER_PATIENCE, factor=0.5, min_lr=1e-8)

        best_train_contig = float('inf')

        for epoch in range(args.epochs):
            avg_train, train_ce, train_dice, train_sobel, train_contig = train_unsupervised(model, teacher_model, train_loader, optimizer, criterion, device, epoch, args.epochs)
            avg_val, avg_ce, avg_dice, avg_sobel, avg_contig = validate_unsupervised(model, val_loader, criterion, device)

            lr_head = optimizer.param_groups[0]['lr']
            lr_info = f"LR head: {lr_head:.2e}"
            
            if len(optimizer.param_groups) > 1:
                lr_info += f" | LR backbone: {optimizer.param_groups[1]['lr']:.2e}"

            print(
                f"Epoch {epoch+1:3d} | {lr_info} | "
                f"Train={avg_train:.4f} (CE={train_ce:.4f}, Dice={train_dice:.4f}, Sobel={train_sobel:.4f}, Contig={train_contig:.4f}) | "
                f"Val={avg_val:.4f} | "
                f"Val_CE={avg_ce:.4f} | Val_Dice={avg_dice:.4f} | "
                f"Val_Sobel={avg_sobel:.4f} | Val_Contig={avg_contig:.4f}"
            )

            scheduler.step(avg_val)

            if train_contig < best_train_contig:
                best_train_contig = train_contig
                torch.save(model.state_dict(), FINETUNE_WEIGHTS_PATH)

            visualize_predictions(model, cnv_vis_ds, device, args.batch_size, num_workers=args.num_workers, teacher_model=teacher_model)

if __name__ == '__main__':
    main()
