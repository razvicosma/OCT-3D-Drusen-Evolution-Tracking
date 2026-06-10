import os
import argparse
from dotenv import load_dotenv

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from scripts.resunet.model import ResUNet
from scripts.resunet.dataset import OCTDataset
from scripts.resunet.flip_loss import compute_ldrflip, color_space_transform
from scripts.resunet.config import WEIGHTS_PATH, PLOTS_DIR, HEATMAPS_PLOT_PATH, PIXELS_PER_DEGREE

def visualize_heatmaps(model, dataset, device, n=3):

    model.eval()
    fig, axes = plt.subplots(n, 5, figsize=(20, 4*n))
    

    with torch.no_grad():
        for i in range(n):

            stripped, original, mask = dataset[i]
            mask = mask.to(device)
            stripped = stripped.to(device)
            
            inp = torch.cat([stripped.unsqueeze(0), mask.unsqueeze(0)], dim=1)
            pred = model(inp)
            
            test = pred.repeat(1, 3, 1, 1)
            reference = original.repeat(3, 1, 1).unsqueeze(0).to(device)
            
            test = torch.clamp(test, 0, 1)
            reference = torch.clamp(reference, 0, 1)
            
            test_opponent = color_space_transform(test, 'srgb2ycxcz')
            reference_opponent = color_space_transform(reference, 'srgb2ycxcz')
            
            deltaE = compute_ldrflip(test_opponent, reference_opponent, PIXELS_PER_DEGREE, 0.7, 0.5, 0.4, 0.95, 1e-15)
            flip_map = deltaE[0, 0].cpu().numpy()
            
            pred_np = pred[0, 0].cpu().numpy()
            orig_np = original[0].cpu().numpy()
            _, ssim_map = ssim(pred_np, orig_np, full=True, data_range=1.0)
            ssim_error_map = 1.0 - ssim_map
            
            axes[i, 0].imshow(stripped[0].cpu().numpy(), cmap="gray")
            axes[i, 0].set_title("Stripped Input")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(orig_np, cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")
            
            axes[i, 2].imshow(pred_np, cmap="gray")
            axes[i, 2].set_title("Reconstruction")
            axes[i, 2].axis("off")
            
            im_flip = axes[i, 3].imshow(flip_map, cmap="magma", vmin=0, vmax=1)
            axes[i, 3].set_title("FLIP Error Map")
            axes[i, 3].axis("off")
            fig.colorbar(im_flip, ax=axes[i, 3], fraction=0.046, pad=0.04)
            
            im_ssim = axes[i, 4].imshow(ssim_error_map, cmap="magma", vmin=0, vmax=1)
            axes[i, 4].set_title("1 - SSIM Map")
            axes[i, 4].axis("off")
            fig.colorbar(im_ssim, ax=axes[i, 4], fraction=0.046, pad=0.04)
            
    plt.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(HEATMAPS_PLOT_PATH, dpi=150)
    plt.close()

def main():
    
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=os.environ.get("RESUNET_DATA_ROOT"))
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print("Loading model")
    model = ResUNet().to(device)
    
    weights_path = WEIGHTS_PATH
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    
    print("Loading dataset")
    test_data_root = os.path.join(args.data_root, "img_test")
    dataset = OCTDataset(test_data_root, max_samples=100)
    
    print("Generating heatmaps")
    visualize_heatmaps(model, dataset, device, n=3)
    print(f"Saved {HEATMAPS_PLOT_PATH}")
    

if __name__ == "__main__":
    main()

