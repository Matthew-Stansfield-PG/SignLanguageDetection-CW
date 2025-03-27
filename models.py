import torch.nn as nn
import torch
import yaml


class Generator(nn.Module):
    def __init__(self):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        super(Generator, self).__init__()
        self.nz = config['nz'] #latent vector size
        self.ngf = config['image_size'] #feature_map_size_generator
        self.noc = config['noc']
        self.main = nn.Sequential(
            # input is Z, going into a convolution

            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            #nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # state size. 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Discriminator takes an 'image': object dimensionality batch_size x 3 x H x W
class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        super(Discriminator, self).__init__()
        self.ndf = config['image_size']#feature map size discriminator
        self.noc = 3
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.noc, self.ndf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ndf * 2),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ndf * 4),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(self.ndf * 8),
            nn.ReLU(True),
            #nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 11 * 11, 1),
            # nn.Sigmoid()
        )

    def forward(self, input):
        features = self.main(input)
        return features

