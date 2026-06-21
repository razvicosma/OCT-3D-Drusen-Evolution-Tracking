import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-5):

        super().__init__()
        self.smooth = smooth

    def forward(self, logits, true_masks):

        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        true_masks_onehot = F.one_hot(true_masks, num_classes=num_classes).permute(0, 3, 1, 2).to(torch.float32)

        dims = (2, 3)
        intersection = torch.sum(probs * true_masks_onehot, dims)
        cardinality = torch.sum(probs + true_masks_onehot, dims)

        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_score.mean()

class SobelEdgeLoss(nn.Module):

    def __init__(self):

        super().__init__()
        sobel_x = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)

    def forward(self, logits, _masks=None):

        probs = F.softmax(logits, dim=1)
        B, C, H, W = probs.shape
        kernel = self.sobel_x.to(probs.device).expand(C, 1, 3, 3)
        edges = F.conv2d(probs, kernel, padding=1, groups=C)

        return edges.abs().mean()

class ColumnContiguityLoss(nn.Module):

    def __init__(self, min_tv=2.0):

        super().__init__()
        self.min_tv = min_tv

    def forward(self, logits, _masks=None):

        probs = F.softmax(logits, dim=1)
        tv = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).sum(dim=2)
        penalty = torch.zeros_like(tv)

        penalty[:, 0, :] = torch.abs(tv[:, 0, :] - 1.0)
        if probs.shape[1] > 5:
            penalty[:, 5, :] = torch.abs(tv[:, 5, :] - 1.0)

        inner_classes = [c for c in range(1, probs.shape[1]) if c != 5]
        if inner_classes:
            penalty[:, inner_classes, :] = torch.abs(tv[:, inner_classes, :] - 2.0)

        return penalty.mean()

class CombinedLoss(nn.Module):

    def __init__(self, weight_ce=0.5, weight_dice=0.5, weight_sobel=0.0, weight_contiguity=0.0, class_weights=None):

        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()
        self.sobel = SobelEdgeLoss()
        self.contiguity = ColumnContiguityLoss()

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_sobel = weight_sobel
        self.weight_contiguity = weight_contiguity

    def forward(self, logits, masks):

        loss = self.weight_ce * self.ce(logits, masks)
        loss += self.weight_dice * self.dice(logits, masks)

        if self.weight_sobel > 0:
            loss += self.weight_sobel * self.sobel(logits)

        if self.weight_contiguity > 0:
            loss += self.weight_contiguity * self.contiguity(logits)

        return loss