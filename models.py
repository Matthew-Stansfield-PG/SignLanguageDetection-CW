import torch.nn as nn
import torch
import yaml

class Generator(nn.Module):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        nz=config['nz']
        ngf=config['ngf']
        nc=config['nc']
    def __init__(self, nz=nz, ngf=ngf, nc=nc, n_classes=3):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, nz)
        self.main = nn.Sequential(
            # Input: Z latent vector
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # Output is normalized [-1, 1]
        )

    def forward(self, input, labels):
        label_embeddings = self.label_embedding(labels)
        input = input + label_embeddings.unsqueeze(2).unsqueeze(3)
        return self.main(input)

class Discriminator(nn.Module):
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        nc=config['nc']
        ndf=config['ndf']
    def __init__(self, nc=nc, ndf=ndf, n_classes=3):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, 64 * 64)
        self.nc=nc
        self.main = nn.Sequential(
            # Input: Image
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),#nc+1 to incorporae label info
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # No Sigmoid!
        )

    def forward(self, input, labels):
        label_map = self.label_embedding(labels).view(labels.size(0), 1, 64, 64)
        input = torch.cat([input, label_map], 1)
        return self.main(input)