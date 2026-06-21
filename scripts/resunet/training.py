import torch
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim

from scripts.resunet.config import ALPHA, PIXELS_PER_DEGREE

def train(model, loader, optimizer, criterion, criterion2, device, alpha=ALPHA):

    model.train()
    total_loss = 0

    for stripped, original, mask in tqdm(loader, desc="Training"):
        stripped = stripped.to(device)
        original = original.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        inp = torch.cat([stripped, mask], dim=1)
        pred = model(inp)

        loss = (1-alpha)*criterion(pred, original) + alpha*criterion2(pred.expand(-1, 3, -1, -1), original.expand(-1, 3, -1, -1),pixels_per_degree=PIXELS_PER_DEGREE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, loader, criterion, criterion2, device, alpha=ALPHA):

    model.eval()
    total_loss = 0
    total_ssim = 0
    total_ms_ssim = 0
    total_ldr_loss = 0

    with torch.no_grad():
        for stripped, original, mask in tqdm(loader, desc="Validating"):
            stripped = stripped.to(device)
            original = original.to(device)
            mask = mask.to(device)

            inp = torch.cat([stripped, mask], dim=1)
            pred = model(inp)
            ldr_loss = criterion2(pred.expand(-1, 3, -1, -1), original.expand(-1, 3, -1, -1),pixels_per_degree=PIXELS_PER_DEGREE)
            loss = (1-alpha)*criterion(pred, original) + alpha*ldr_loss
            total_loss += loss.item()
            total_ldr_loss += ldr_loss.item()

            ms_ssim_val = ms_ssim(pred, original, data_range=1.0, size_average=True)
            total_ms_ssim += ms_ssim_val.item()

            ssim_val = ssim(pred, original, data_range=1.0, size_average=True)
            total_ssim += ssim_val.item()

    return total_loss / len(loader), total_ssim / len(loader), total_ms_ssim / len(loader), total_ldr_loss / len(loader)

