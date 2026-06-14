import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from scripts.dino.config import NUM_CLASSES

def compute_class_weights(dataset, batch_size, num_workers, device):

    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_pixels = 0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    for _, masks in tqdm(loader, desc="Computing Weights"):
        bincount = torch.bincount(masks.flatten(), minlength=NUM_CLASSES)
        class_counts += bincount.numpy()
        total_pixels += masks.numel()

    class_weights = total_pixels / (NUM_CLASSES * np.maximum(class_counts, 1))
    class_weights = torch.from_numpy(class_weights).to(torch.float32).to(device)

    return class_weights

def build_optimizer(model, args):

    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = [p for name, p in model.named_parameters() if 'backbone' not in name and p.requires_grad]
    param_groups = [{'params': head_params, 'lr': args.lr, 'name': 'head'}]

    n_unfreeze = args.unfreeze_blocks

    if n_unfreeze != 0:
        if n_unfreeze == -1:
            for param in model.backbone.parameters():
                param.requires_grad = True
            tag = "entire backbone"
        else:
            all_blocks = list(model.backbone.blocks)
            for block in all_blocks[-n_unfreeze:]:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(model.backbone, 'norm'):
                for param in model.backbone.norm.parameters():
                    param.requires_grad = True
            tag = f"last {n_unfreeze} block(s) + norm"

        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        param_groups.append({'params': backbone_params, 'lr': args.lr / 10.0, 'name': 'backbone_tail'})
        print(f"Backbone unfrozen ({tag}) | head LR={args.lr:.2e} | backbone LR={args.lr/10:.2e}")
    else:
        print(f"Backbone frozen | head LR={args.lr:.2e}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    return optimizer

def train(model, loader, optimizer, criterion, device, epoch, epochs):

    model.train()
    train_loss = 0

    for images, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(loader)

def validate(model, loader, criterion, device):

    model.eval()
    val_loss = 0
    val_ce = 0
    val_dice = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            ce_loss = criterion.ce(outputs, masks)
            dice_loss = criterion.dice(outputs, masks)

            val_loss += loss.item()
            val_ce += ce_loss.item()
            val_dice += dice_loss.item()

    n = len(loader)

    return val_loss / n, val_ce / n, val_dice / n

def validate_unsupervised(model, loader, criterion, device):

    model.eval()
    val_loss = 0
    val_ce = 0
    val_dice = 0
    val_sobel = 0
    val_contiguity = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating (Unsupervised)"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            loss = criterion(outputs, masks)
            ce_loss = criterion.ce(outputs, masks)
            dice_loss = criterion.dice(outputs, masks)
            sobel_loss = criterion.sobel(outputs)
            contiguity_loss = criterion.contiguity(outputs)

            val_loss += loss.item()
            val_ce += ce_loss.item()
            val_dice += dice_loss.item()
            val_sobel += sobel_loss.item()
            val_contiguity += contiguity_loss.item()

    n = len(loader)

    return val_loss / n, val_ce / n, val_dice / n, val_sobel / n, val_contiguity / n

def train_unsupervised(student_model, teacher_model, loader, optimizer, criterion, device, epoch, epochs):

    student_model.train()
    teacher_model.eval()
    
    total_loss = 0.0
    total_ce = 0.0
    total_dice = 0.0
    total_sobel = 0.0
    total_contig = 0.0

    for images, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images = images.to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(images)
            pseudo_masks = torch.argmax(teacher_logits, dim=1)

        optimizer.zero_grad()

        outputs = student_model(images)
        loss = criterion(outputs, pseudo_masks)
        
        ce_loss = criterion.ce(outputs, pseudo_masks)
        dice_loss = criterion.dice(outputs, pseudo_masks)
        sobel_loss = criterion.sobel(outputs)
        contig_loss = criterion.contiguity(outputs)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_dice += dice_loss.item()
        total_sobel += sobel_loss.item()
        total_contig += contig_loss.item()

    n = len(loader)
    return total_loss / n, total_ce / n, total_dice / n, total_sobel / n, total_contig / n
