import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

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
    total_ldr_loss = 0
    n_samples = 0

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

            pred_np = pred.cpu().numpy()[:, 0]
            orig_np = original.cpu().numpy()[:, 0]

            for p, o in zip(pred_np, orig_np):
                total_ssim += ssim(p, o, data_range=1.0)
                n_samples += 1

    return total_loss / len(loader), total_ssim / n_samples, total_ldr_loss / len(loader)

