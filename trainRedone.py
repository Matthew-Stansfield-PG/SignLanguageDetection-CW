import os
import numpy as np
import random
import torch
from torch import nn, optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import yaml
import time
import math
import multiprocessing
import wandb

from dataset import get_data
from models import Discriminator, Generator
from logger import Logger


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == '__main__':
    multiprocessing.freeze_support()

    manualSeed = 42
    print("Random Seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device:", device)

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    training_set = get_data()
    batch_size = config['batch_size']
    generator_input_size = config['nz']
    num_epochs = config['epochs']
    learning_rate = config['lr']
    beta1 = 0.5
    sample_size = len(training_set) // batch_size * batch_size
    indices_list = list(range(sample_size))
    training_set = torch.utils.data.Subset(training_set, indices_list)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)

    denormalize = transforms.Compose([
        transforms.Normalize((0, 0, 0), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
        transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1))
    ])

    for real_samples, real_label in train_loader:
        for i in range(batch_size):
            ax = plt.subplot(math.ceil(math.sqrt(batch_size)), math.ceil(math.sqrt(batch_size)), i + 1)
            sample = denormalize(real_samples[i])
            plt.imshow(sample.permute(1, 2, 0))
            plt.xticks([]); plt.yticks([])
        break

    generator = Generator(config['nz'], 64, 3).to(device)
    discriminator = Discriminator(3, 64).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)


    loss_function = nn.BCEWithLogitsLoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    logger = Logger('gan-training').get_logger()


    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    fixed_noise = torch.randn(64, generator_input_size, 1, 1).to(device)

    for epoch in range(num_epochs + 1):
        start = time.time()
        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            b_size = real_samples.size(0)

            # Smoothing: real=0.9, fake=0.0
            real_labels = torch.full((b_size,), 0.9, dtype=torch.float, device=device)
            fake_labels = torch.zeros(b_size, dtype=torch.float, device=device)

            # Train Discriminator
            discriminator.zero_grad()

            output_real = discriminator(real_samples).view(-1)
            loss_real = loss_function(output_real, real_labels)

            noise = torch.randn(b_size, generator_input_size, 1, 1, device=device)
            fake_samples = generator(noise)
            output_fake = discriminator(fake_samples.detach()).view(-1)
            loss_fake = loss_function(output_fake, fake_labels)

            loss_discriminator = loss_real + loss_fake
            loss_discriminator.backward()
            optimizer_discriminator.step()

            generator.zero_grad()
            output = discriminator(fake_samples).view(-1)
            loss_generator = loss_function(output, real_labels)
            loss_generator.backward()
            optimizer_generator.step()

            if n % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {n}/{len(train_loader)} \
                      Loss D: {loss_discriminator.item():.4f}, loss G: {loss_generator.item():.4f}")

        logger.log({
            'epoch': epoch,
            'loss_discriminator': loss_discriminator.item(),
            'loss_generator': loss_generator.item()
        }, step=epoch)

        print("Time for epoch:", round(time.time() - start, 2), "seconds")

        if epoch % 5 == 0:
            generator.eval()
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
                fake_images = (fake_images + 1) / 2
                vutils.save_image(fake_images, f"{output_dir}/epoch_{epoch}.png", nrow=8, normalize=True)
                logger.log({
                    "DCGAN Generated Images": wandb.Image(vutils.make_grid(fake_images, nrow=8).permute(1,2,0).numpy(), caption=f"Epoch: {epoch}")
                }, step=epoch)#assembles images in a grid, permute swaps order of channels, height, width to h, w, c for wandb's structure
            generator.train()
            print("saving and logging images...")

    torch.save(generator.state_dict(), "animalImageGAN.pt")
    torch.save(generator, "animalImageGAN_full.pt")
