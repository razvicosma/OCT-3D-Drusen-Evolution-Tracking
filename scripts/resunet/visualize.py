import os

import torch
import matplotlib.pyplot as plt

from scripts.resunet.config import PLOTS_DIR, RECONSTRUCTION_PLOT_PATH
from scripts.resunet.training import extract_cls_dino_feature, extract_and_resize_valid_columns


def visualize_results(model, dino_model, dataset, device, n=3):

    model.eval()
    dino_model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    
    with torch.no_grad():
        for i in range(n):
            stripped, original, mask = dataset[i]
            mask = mask.to(device)
            stripped = stripped.to(device)
            
            inp = torch.cat([stripped.unsqueeze(0), mask.unsqueeze(0)], dim=1)
            
            dino_input = extract_and_resize_valid_columns(stripped.unsqueeze(0), mask.unsqueeze(0), target_size=(256, 256))
            in_c_rgb = dino_input.repeat(1, 3, 1, 1)
            dino_embeds = extract_cls_dino_feature(dino_model, in_c_rgb)

            pred = model(inp, dino_embeds)
            
            pred_np = pred[0, 0].cpu().numpy()
            orig_np = original.numpy()[0]
            
            axes[i, 0].imshow(stripped[0].cpu().numpy(), cmap="gray")
            axes[i, 0].set_title("Stripped Input")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(orig_np, cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(pred_np, cmap="gray")
            axes[i, 2].set_title("Reconstruction")
            axes[i, 2].axis("off")
    
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(RECONSTRUCTION_PLOT_PATH, dpi=150)
    plt.close()
