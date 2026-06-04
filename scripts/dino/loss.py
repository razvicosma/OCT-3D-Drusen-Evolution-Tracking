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

class CombinedLoss(nn.Module):

    def __init__(self, weight_ce=1.0, weight_dice=1.0, class_weights=None):

        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        
    def forward(self, logits, masks):

        return self.weight_ce * self.ce(logits, masks) + self.weight_dice * self.dice(logits, masks)
