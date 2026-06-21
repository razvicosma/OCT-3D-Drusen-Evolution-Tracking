import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from scripts.dino.config import (
    IMAGENET_MEAN, IMAGENET_STD, NUM_CLASSES,
    PLOTS_DIR, CONFUSION_MATRIX_PLOT_PATH, PREDICTIONS_PLOT_PATH
)

def plot_confusion_matrix(model, dataset, device, batch_size, num_workers=4):

    model.eval()

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            masks = masks.numpy()

            for i in range(len(masks)):
                y_true = masks[i].flatten()
                y_pred = preds[i].flatten()
                valid = (y_true >= 0) & (y_true < NUM_CLASSES)
                cm += np.bincount(NUM_CLASSES * y_true[valid] + y_pred[valid], minlength=NUM_CLASSES*NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)

    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.figure.colorbar(im, ax=ax_cm)
    classes = [str(i) for i in range(NUM_CLASSES)]

    ax_cm.set(xticks=np.arange(cm.shape[1]),
              yticks=np.arange(cm.shape[0]),
              xticklabels=classes, yticklabels=classes,
              title='Confusion Matrix',
              ylabel='True label',
              xlabel='Predicted label')

    thresh = cm.max() / 2. if cm.max() > 0 else 1.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, f"{cm[i, j]:,d}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    fig_cm.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig_cm.savefig(CONFUSION_MATRIX_PLOT_PATH)
    plt.close(fig_cm)


def plot_sample_predictions(model, dataset, device, num_samples=3, teacher_model=None):

    model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    indices = random.sample(range(len(dataset)), num_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]

            img_tensor = image.unsqueeze(0).to(device)
            logits = model(img_tensor)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            display_img = image.permute(1, 2, 0).numpy()
            display_img = (display_img * IMAGENET_STD + IMAGENET_MEAN).clip(0, 1)

            if teacher_model is not None:
                teacher_logits = teacher_model(img_tensor)
                middle_mask = torch.argmax(teacher_logits, dim=1).squeeze(0).cpu().numpy()
                middle_title = "Teacher Pseudo-Labels"
                pred_title = "Student Prediction"
            else:
                middle_mask = mask.numpy()
                middle_title = "Ground Truth Mask"
                pred_title = "Predicted Mask"

            axes[i, 0].imshow(display_img, cmap='gray')
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(middle_mask, cmap='tab10', vmin=0, vmax=NUM_CLASSES - 1)
            axes[i, 1].set_title(middle_title)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=NUM_CLASSES - 1)
            axes[i, 2].set_title(pred_title)
            axes[i, 2].axis('off')

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(PREDICTIONS_PLOT_PATH)
    plt.close()


def visualize_predictions(model, dataset, device, batch_size, num_workers=4, num_samples=3, teacher_model=None):

    plot_confusion_matrix(model, dataset, device, batch_size, num_workers)
    plot_sample_predictions(model, dataset, device, num_samples, teacher_model=teacher_model)
