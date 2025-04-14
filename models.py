import torch.nn as nn
import torch
import yaml

class Generator(nn.Module):
    def __init__(self):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        super(Generator, self).__init__()
        self.nz = config['nz']  # latent vector size
        self.ngf = config['ngf']  # feature_map_size_generator
        self.noc = config['noc']  # number of output channels (e.g., 3 for RGB)

        self.main = nn.Sequential(
            # Latent space to initial image size (4x4)
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0),  # 4x4
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(),

            # Upscaling to 8x8
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(),

            # Upscaling to 16x16
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(),

            # Upscaling to 32x32
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(),

            # Upscaling to 64x64
            nn.ConvTranspose2d(self.ngf, self.noc, 4, 2, 1),  # 64x64
            nn.Tanh()  # Output image in the range [-1, 1]
        )

    def forward(self, x):
        output = self.main(x)
        return self.main(x)

# Discriminator for 64x64 images
class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        super(Discriminator, self).__init__()
        self.ndf = config['ndf']  # feature map size discriminator
        self.noc = 3  # Number of output channels (e.g., RGB for color images)

        self.main = nn.Sequential(
            # Input is (noc) x 64 x 64
            nn.Conv2d(self.noc, self.ndf, 4, 2, 1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(True),
            nn.Dropout(0.3),

            # State size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),  # 32x32 -> 16x16
            #nn.BatchNorm2d(self.ndf * 2),
            nn.ReLU(True),
            nn.Dropout(0.3),

            # State size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),  # 16x16 -> 8x8
            #nn.BatchNorm2d(self.ndf*4),
            nn.ReLU(True),
            nn.Dropout(0.3),

            # State size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),  # 8x8 -> 4x4
            #nn.BatchNorm2d(self.ndf * 8),
            nn.ReLU(True),
            nn.Dropout(0.3),

            # State size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),  # 4x4 -> 1x1
            nn.Sigmoid()  # Output probability for real/fake
        )

    def forward(self, x):
        # features = self.main(x)
        output = self.main(x)
        # return features.view(input.size(0), -1).mean(dim=1, keepdim=True)
        return output
