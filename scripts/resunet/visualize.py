import os

import torch
import matplotlib.pyplot as plt

from scripts.resunet.config import PLOTS_DIR, RECONSTRUCTION_PLOT_PATH


def visualize_results(model, dino_feat, dataset, device, n=3):

    model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    
    with torch.no_grad():
        for i in range(n):
            stripped, original, mask = dataset[i]
            mask = mask.to(device)
            stripped = stripped.to(device)
            
            inp = torch.cat([stripped.unsqueeze(0), mask.unsqueeze(0)], dim=1)
            pred = model(inp, dino_feat.expand(1, -1, -1, -1))
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
