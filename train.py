import os
import numpy as np
import pandas as pd
import torch
import random
from torch import nn, optim
import torchvision.transforms as transforms
import time
import multiprocessing
from dataset import get_data
from models import Discriminator, Generator
from logger import Logger
import torchvision.utils as vutils
import os
import yaml
if __name__ == '__main__':
    multiprocessing.freeze_support()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("Current device:", device)
    manualSeed = 42
    print("Random Seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)
    num_epochs = 500
    batch_size = 10
    generator_input_size = config['nz']
    learning_rate = 0.0002
    beta1 = 0.5
    # Get dataset and DataLoader
    training_set = get_data()
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    # Loss and optimizers
    loss_function = nn.BCELoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    # Logger
    logger = Logger('gan-training').get_logger()
    # Create output directory
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    # Fixed noise for evaluation (e.g., 64 samples)
    fixed_noise = torch.randn(64, generator_input_size, 1, 1).to(device)
    start = time.time()
    for epoch in range(num_epochs + 1):
        print("Running epoch number: ",epoch)
        for n, (real_samples, _) in enumerate(train_loader):
            print(n)
            real_samples = real_samples.to(device)
            real_samples_labels = torch.ones((real_samples.size(0), 1)).to(device)
            # Generate fake samples
            latent_space_samples = torch.randn((real_samples.size(0), generator_input_size, 1, 1)).to(device)
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((real_samples.size(0), 1)).to(device)
            # Train Discriminator
            discriminator.zero_grad()
            all_samples = torch.cat((real_samples, generated_samples))
            all_labels = torch.cat((real_samples_labels, generated_samples_labels))
            output_discriminator = discriminator(all_samples).view(-1, 1)
            loss_discriminator = loss_function(output_discriminator, all_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()
            # Train Generator
            generator.zero_grad()
            latent_space_samples = torch.randn((real_samples.size(0), generator_input_size, 1, 1)).to(device)
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples).view(-1, 1)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()
        print("epoch training ended")
        print(epoch % 5)
        if epoch % 5 == 0:
            print("Time taken for 5 epochs: {:.2f}s".format(time.time() - start))
            start = time.time()
            print(f"Epoch {epoch} - Loss D: {loss_discriminator.item():.4f}, Loss G: {loss_generator.item():.4f}")
            logger.log({
                'epoch': epoch,
                'loss_discriminator': loss_discriminator.item(),
                'loss_generator': loss_generator.item()
            })
            # Save generated images
            generator.eval()
            with torch.no_grad():
                fake_images = generator(fixed_noise)
                fake_images = (fake_images + 1) / 2  # Rescale from [-1, 1] to [0, 1]
                vutils.save_image(fake_images, os.path.join(output_dir, f"epoch_{epoch}.png"), nrow=8, normalize=True)
            generator.train()
        print("epoch ended")