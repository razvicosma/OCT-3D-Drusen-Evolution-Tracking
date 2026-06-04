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

    def __init__(self, in_ch, out_ch, skip_ch):

        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv_in = nn.Conv2d(out_ch + skip_ch, out_ch, 1, bias=False)
        self.resblock = ResBlock(out_ch)

    def forward(self, x, skip):

        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_in(x)

        return self.resblock(x)


class ResUNet(nn.Module):

    def __init__(self, base_ch=64):

        super().__init__()

        self.enc1 = EncoderBlock(2, base_ch)
        self.enc2 = EncoderBlock(base_ch, base_ch*2)
        self.enc3 = EncoderBlock(base_ch*2, base_ch*4)
        self.enc4 = EncoderBlock(base_ch*4, base_ch*8)
        self.enc5 = EncoderBlock(base_ch*8, base_ch*16)

        self.bottleneck = ResBlock(base_ch*16)

        self.dec5 = DecoderBlock(base_ch*16, base_ch*8, base_ch*16)
        self.dec4 = DecoderBlock(base_ch*8, base_ch*4, base_ch*8)
        self.dec3 = DecoderBlock(base_ch*4, base_ch*2, base_ch*4)
        self.dec2 = DecoderBlock(base_ch*2, base_ch, base_ch*2)
        self.dec1 = DecoderBlock(base_ch, base_ch//2, base_ch)

        self.out = nn.Conv2d(base_ch//2, 1, 1)

    def forward(self, x):

        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x, s5 = self.enc5(x)

        x = self.bottleneck(x)

        x = self.dec5(x, s5)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return torch.sigmoid(self.out(x))