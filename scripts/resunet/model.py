import torch
import torch.nn as nn

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


class FiLM(nn.Module):

    def __init__(self, cond_dim, feature_dim):

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )

    def forward(self, x, cond):

        film_params = self.mlp(cond)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class EncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.resblock = ResBlock(out_ch)
        self.downsample = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv_in(x)
        skip = self.resblock(x)

        return self.downsample(skip), skip


class DecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch, skip_ch, cond_dim):

        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv_in = nn.Conv2d(out_ch + skip_ch, out_ch, 1, bias=False)
        self.resblock = ResBlock(out_ch)
        self.film = FiLM(cond_dim, out_ch)

    def forward(self, x, skip, cond):

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_in(x)
        
        x = self.resblock(x)
        x = self.film(x, cond)

        return x


class ResUNet(nn.Module):

    def __init__(self, base_ch=64, dino_embed_dim=768):

        super().__init__()

        self.enc1 = EncoderBlock(2, base_ch)
        self.enc2 = EncoderBlock(base_ch, base_ch*2)
        self.enc3 = EncoderBlock(base_ch*2, base_ch*4)
        self.enc4 = EncoderBlock(base_ch*4, base_ch*8)
        self.enc5 = EncoderBlock(base_ch*8, base_ch*16)

        self.film = FiLM(cond_dim=dino_embed_dim, feature_dim=base_ch*16)

        self.bottleneck = ResBlock(base_ch*16)

        self.dec5 = DecoderBlock(base_ch*16, base_ch*8, base_ch*16, cond_dim=dino_embed_dim)
        self.dec4 = DecoderBlock(base_ch*8, base_ch*4, base_ch*8, cond_dim=dino_embed_dim)
        self.dec3 = DecoderBlock(base_ch*4, base_ch*2, base_ch*4, cond_dim=dino_embed_dim)
        self.dec2 = DecoderBlock(base_ch*2, base_ch, base_ch*2, cond_dim=dino_embed_dim)
        self.dec1 = DecoderBlock(base_ch, base_ch//2, base_ch, cond_dim=dino_embed_dim)

        self.out = nn.Conv2d(base_ch//2, 1, 1)

    def forward(self, x, dino_embeds):

        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x, s5 = self.enc5(x)

        x = self.bottleneck(x)
        x = self.film(x, dino_embeds)

        x = self.dec5(x, s5, cond=dino_embeds)
        x = self.dec4(x, s4, cond=dino_embeds)
        x = self.dec3(x, s3, cond=dino_embeds)
        x = self.dec2(x, s2, cond=dino_embeds)
        x = self.dec1(x, s1, cond=dino_embeds)

        return torch.sigmoid(self.out(x))