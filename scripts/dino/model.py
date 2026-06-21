import timm
import torch.nn as nn

from scripts.dino.config import BACKBONE_WEIGHTS, IMAGE_SIZE, NUM_CLASSES


class ResBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        return self.relu(self.block(x) + x)


class DecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.resblock = ResBlock(out_ch)

    def forward(self, x):

        return self.resblock(self.upsample(x))


class DINOv3Segmenter(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()
        
        backbone = timm.create_model(
            'vit_small_patch16_dinov3.lvd1689m',
            pretrained=False,
            pretrained_cfg_overlay=dict(file=BACKBONE_WEIGHTS),
            img_size=IMAGE_SIZE
        )
        self.backbone = backbone
        self.embed_dim = self.backbone.embed_dim

        self.dec4 = DecoderBlock(self.embed_dim, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 32)

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):

        B, C, H, W = x.shape
        patch_h, patch_w = self.backbone.patch_embed.patch_size
        H_feat, W_feat = H // patch_h, W // patch_w
        n_prefix = self.backbone.num_prefix_tokens

        features = self.backbone.forward_features(x)
        patch_tok = features[:, n_prefix:]
        x_feat = patch_tok.transpose(1, 2).reshape(B, self.embed_dim, H_feat, W_feat)

        x_feat = self.dec4(x_feat)
        x_feat = self.dec3(x_feat)
        x_feat = self.dec2(x_feat)
        x_feat = self.dec1(x_feat)

        return self.final_conv(x_feat)
