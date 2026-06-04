import os
import argparse
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader

from scripts.dino.loss import CombinedLoss
from scripts.dino.model import DINOv3Segmenter
from scripts.dino.visualize import visualize_predictions
from scripts.dino.dataset import OCTDataset, AugmentedOCTDataset
from scripts.dino.training import build_optimizer, compute_class_weights, train, validate
from scripts.dino.config import NUM_CLASSES, SCHEDULER_PATIENCE, WEIGHTS_PATH

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--data_root", type=str, default=os.environ.get("DINO_DATA_ROOT"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--unfreeze_blocks", type=int, default=5)

    return parser.parse_args()

def main():
    
    load_dotenv()

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config: {args.batch_size} BS | {args.lr} LR | {device}")

    model = DINOv3Segmenter(num_classes=NUM_CLASSES).to(device)

    train_img_dir = os.path.join(args.data_root, "train_img")
    train_msk_dir = os.path.join(args.data_root, "train_msk")
    valid_img_dir = os.path.join(args.data_root, "valid_img")
    valid_msk_dir = os.path.join(args.data_root, "valid_msk")
    test_img_dir = os.path.join(args.data_root, "test_img")
    test_msk_dir = os.path.join(args.data_root, "test_msk")

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
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

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

if __name__ == '__main__':
    main()
