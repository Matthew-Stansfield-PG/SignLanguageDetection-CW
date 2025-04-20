import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import yaml

current_epoch = 0

class ProgressiveSAGANSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.scale = in_channels ** -0.5
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.InstanceNorm2d(in_channels, affine=True)

    def forward(self, x):
        if current_epoch < 150:
            return x

        alpha = min(1.0, max(0.0, (current_epoch - 150) / 150))

        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)

        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)

        out = x + alpha * self.gamma * out
        return self.norm(out)


class Generator(nn.Module):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        nz = config['nz']
        ngf = config['ngf']
        nc = config['nc']

    def __init__(self, nz=nz, ngf=ngf, nc=nc, n_classes=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.label_embedding = nn.Embedding(n_classes, 10)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + 10, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            ProgressiveSAGANSelfAttention(ngf * 2),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input, labels):
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat((input, label_emb), dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        nc = config['nc']
        ndf = config['ndf']

    def __init__(self, nc=nc, ndf=ndf, n_classes=3, embed_dim=1):
        super(Discriminator, self).__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(n_classes, self.embed_dim)

        self.main = nn.Sequential(
            nn.Conv2d(nc + self.embed_dim, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            ProgressiveSAGANSelfAttention(ndf * 4),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input, labels):
        label_map = self.embed(labels).unsqueeze(2).unsqueeze(3)
        label_map = label_map.expand(-1, -1, 64, 64)
        input = torch.cat([input, label_map], dim=1)
        return self.main(input)
