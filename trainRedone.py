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
import math
import matplotlib.pyplot as plt

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

    num_epochs = config['epochs']  # was 500
    batch_size = config['batch_size']
    generator_input_size = config['nz']
    learning_rate = config['lr']
    beta1 = 0.5

    sample_size = len(training_set) // batch_size * batch_size
    print(f"Total Sample Size: {sample_size}")

    indices_list = list(range(0, sample_size))#creates a list of numbers which will be used to select which samples wll be used
    training_set = torch.utils.data.Subset(training_set, indices_list)#creates subset of data points based on the index of that which is in the indices_list
    #print(training_set)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)

    #denormalizes data after it had been normalized for the GAN's Tanh
    transform_denormalize = transforms.Compose(
        [transforms.Normalize((0, 0, 0), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
         transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1))])


    # for n, (real_samples, real_label) in enumerate(train_loader):
    #     for i in range(batch_size):
    #         ax = plt.subplot(math.ceil(math.sqrt(batch_size)), math.ceil(math.sqrt(batch_size)), i + 1)
    #         sample = real_samples[i]
    #         sample = transform_denormalize(sample)
    #         plt.imshow(sample.squeeze().permute(1, 2, 0))
    #         plt.title(real_label[i])
    #         plt.xticks([])
    #         plt.yticks([])
    #     break

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

    fixed_noise = torch.randn(64, generator_input_size, 1, 1).to(device)

    start = time.time()
    for epoch in range(num_epochs + 1):
        for n, (real_samples, _) in enumerate(train_loader):
            if n % 10 == 0:
                print(n)
            real_samples = real_samples.to(device)
            real_samples_labels = torch.ones((batch_size, 1)).to(device)

            latent_space_samples = torch.randn((batch_size, generator_input_size, 1, 1)).to(device)
            generated_samples = generator(latent_space_samples)

            generated_samples_labels = torch.zeros((batch_size, 1)).to(device)
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # Train discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            output_discriminator = output_discriminator.view(output_discriminator.size()[:2])
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Train generator
            generator.zero_grad()
            latent_space_samples = torch.randn((batch_size, generator_input_size, 1, 1)).to(device)
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            output_discriminator_generated = output_discriminator_generated.view(
                output_discriminator_generated.size()[:2])
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()


        print(f"Time taken for epoch: {time.time() - start:.2f}s")
        start = time.time()
        print(f"Epoch: {epoch} Loss D.: {loss_discriminator:.4f}")
        print(f"Epoch: {epoch} Loss G.: {loss_generator:.4f}")

        if epoch % 5 == 0:
            generator.eval()
            with torch.no_grad():

                fake_images = generator(fixed_noise)
                fake_images = (fake_images + 1) / 2
                vutils.save_image(fake_images, os.path.join(output_dir, f"epoch_{epoch}.png"), nrow=8, normalize=True)
                print("saving images...")
            generator.train()
        print("epoch ended")

    torch.save(generator.state_dict(), "animalImageGAN.pt")
    torch.save(generator, "animalImageGAN_full.pt")