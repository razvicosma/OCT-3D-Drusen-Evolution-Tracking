import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

from scripts.resunet.config import ALPHA, PIXELS_PER_DEGREE

def extract_cls_dino_feature(dino_model, x_rgb):

    features = dino_model.forward_features(x_rgb)
    cls_token = features[:, 0, :]
    return cls_token

def extract_and_resize_valid_columns(stripped, mask, target_size=(256, 256)):

    B = stripped.size(0)
    resized_list = []
    for i in range(B):
        img_i = stripped[i, 0]
        mask_i = mask[i, 0]
        
        valid_cols = mask_i[0] > 0.5
        extracted = img_i[:, valid_cols]
        
        if extracted.size(1) == 0:
            extracted = torch.zeros((512, 1), device=stripped.device)
            
        extracted = extracted.unsqueeze(0).unsqueeze(0)
        resized = torch.nn.functional.interpolate(extracted, size=target_size, mode='bilinear', align_corners=False)
        resized_list.append(resized.squeeze(0))
        
    return torch.stack(resized_list)

def train(model, dino_model, loader, optimizer, criterion, criterion2, device, alpha=ALPHA):

    model.train()
    dino_model.eval()
    total_loss = 0

    for stripped, original, mask in tqdm(loader, desc="Training"):
        stripped = stripped.to(device)
        original = original.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        inp = torch.cat([stripped, mask], dim=1)
        
        with torch.no_grad():
            dino_input = extract_and_resize_valid_columns(stripped, mask, target_size=(256, 256))
            in_c_rgb = dino_input.repeat(1, 3, 1, 1)
            dino_embeds = extract_cls_dino_feature(dino_model, in_c_rgb)

        pred = model(inp, dino_embeds)
        
        loss = (1-alpha)*criterion(pred, original) + alpha*criterion2(pred.expand(-1, 3, -1, -1), original.expand(-1, 3, -1, -1),pixels_per_degree=PIXELS_PER_DEGREE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def validate(model, dino_model, loader, criterion, criterion2, device, alpha=ALPHA):

    model.eval()
    dino_model.eval()
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
            
            dino_input = extract_and_resize_valid_columns(stripped, mask, target_size=(256, 256))
            in_c_rgb = dino_input.repeat(1, 3, 1, 1)
            dino_embeds = extract_cls_dino_feature(dino_model, in_c_rgb)

            pred = model(inp, dino_embeds)
            
            l1_loss = criterion(pred, original)
            ldr_loss = criterion2(pred.expand(-1, 3, -1, -1), original.expand(-1, 3, -1, -1), pixels_per_degree=PIXELS_PER_DEGREE)
            
            loss = (1-alpha)*l1_loss + alpha*ldr_loss
            
            total_loss += loss.item()
            total_ldr_loss += ldr_loss.item()

            pred_np = pred.cpu().numpy()[:, 0]
            orig_np = original.cpu().numpy()[:, 0]

            for p, o in zip(pred_np, orig_np):
                total_ssim += ssim(p, o, data_range=1.0)
                n_samples += 1

    return total_loss / len(loader), total_ssim / n_samples, total_ldr_loss / len(loader)

