import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils import spectral_norm
import torch.nn as nn
import torch.nn.functional as F

current_epoch = 0
class ProgressiveSAGANSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=2):
        super().__init__()
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)
    def forward(self, x):
        if current_epoch < 100:
            return x
        alpha = min(1.0, max(0.0, (current_epoch - 100) / 100))
        B, C, H, W = x.shape
        N = H * W
        head_dim = self.head_dim
        q = self.query(x).reshape(B, self.num_heads, head_dim, N).permute(0, 1, 3, 2)
        k = self.key(x).reshape(B, self.num_heads, head_dim, N).permute(0, 1, 3, 2)
        v = self.value(x).reshape(B, self.num_heads, head_dim, N).permute(0, 1, 3, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = x + alpha * self.gamma * out
        return self.norm(out)
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, n_classes=3, resolution=64):
        super(Generator, self).__init__()
        self.resolution = resolution
        self.label_embed = nn.Embedding(n_classes, 10)

        # Initial block
        self.init_block = nn.Sequential(
            nn.ConvTranspose2d(nz + 10, ngf * 8, 4, 1, 0, bias=False),
            ProgressiveSAGANSelfAttention(ngf * 8),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True))

        # 64x64 blocks
        self.block_64 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            ProgressiveSAGANSelfAttention(ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            ProgressiveSAGANSelfAttention(ngf * 2),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        self.to_rgb_64 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # 128x128 blocks
        self.block_128 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            # ProgressiveSAGANSelfAttention(ngf),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            # ProgressiveSAGANSelfAttention(ngf),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
        )

        self.to_rgb_128 = nn.Sequential(
            nn.Conv2d(ngf, nc, kernel_size=3, padding=1),
            nn.Tanh())

        # 256x256 blocks
        self.block_256 = nn.Sequential(
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        self.to_rgb_256 = nn.Sequential(
            nn.Conv2d(ngf, nc, kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, input, labels, alpha=1.0):
        label_emb = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat((input, label_emb), dim=1)
        x = self.init_block(x)
        x = self.block_64(x)
        if self.resolution >= 256:
            if alpha < 1.0:  # During fade-in
                img_128 = self.block_128(x)
                img_128 = self.to_rgb_128(x)
            print("got into 256")
            x = self.block_128(x)
            x = self.block_256(x)
            img_256 = self.to_rgb_256(x)
            if alpha < 1.0:
                img_128_upsampled = F.interpolate(img_128, scale_factor=2, mode='bilinear', align_corners=False)
                out = alpha * img_256 + (1 - alpha) * img_128_upsampled
                return out
            else:
                return img_256
        elif self.resolution >= 128:
            if alpha < 1.0:  # During fade-in
                img_64 = self.to_rgb_64(x)
            x = self.block_128(x)
            img_128 = self.to_rgb_128(x)
            print(x.shape)

            if alpha < 1.0:
                img_64_upsampled = F.interpolate(img_64, scale_factor=2, mode='bilinear', align_corners=False)
                out = alpha * img_128 + (1 - alpha) * img_64_upsampled
                return out
            else:
                return img_128
        else:
            return self.to_rgb_64(x)


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64, n_classes=3, resolution=64):
        super(Discriminator, self).__init__()
        self.resolution = resolution
        self.label_embed = nn.Embedding(n_classes, 10)

        # Common blocks
        self.block1 = nn.Sequential(
            nn.Conv2d(nc + 10, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2))

        self.block2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            ProgressiveSAGANSelfAttention(ndf * 2),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2))

        self.block3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            ProgressiveSAGANSelfAttention(ndf * 4),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2))

        self.block4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2))
        self.block5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2))
        # Resolution-specific final blocks
        if resolution >= 256:
            self.final_256 = nn.Sequential(
                nn.Conv2d(ndf * 16, 1, 4, 1, 0))
        elif resolution >= 128:
            self.final_128 = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, 4, 1, 0))
        else:
            self.final_64 = nn.Sequential(
                nn.Conv2d(ndf * 4, 1, 4, 1, 0))

    def forward(self, x, labels, alpha=1.0):
        # Process labels
        label_emb = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat((x, label_emb), dim=1)

        # Handle resolution transition
        if self.resolution >= 128 and x.size(2) == 64:
            x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = self.block1(x)
        x = self.block2(x)

        if self.resolution >= 256:
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.final_256(x)
        elif self.resolution >= 128:
            x = self.block3(x)
            x = self.block4(x)
            x = self.final_128(x)
        else:
            x = self.block3(x)
            assert x.size(2) >= 4 and x.size(3) >= 4, f"Feature map too small: {x.shape}"
            x = self.final_64(x)
        return x.view(-1)