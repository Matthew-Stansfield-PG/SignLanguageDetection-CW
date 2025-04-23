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
import os

from dataset import get_data
from models import Discriminator, Generator
from logger import Logger
from torchmetrics.image.fid import FrechetInceptionDistance

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

fid_metric = FrechetInceptionDistance(feature=2048).to('cuda' if torch.cuda.is_available() else 'cpu')
resize_for_fid = transforms.Resize((299, 299))

def to_uint8(tensor):
    tensor = (tensor * 127.5 + 127.5).clamp(0, 255)
    return tensor.to(torch.uint8)

def compute_fid_score(generator, real_images, labels, device, nz):
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    generator.eval()
    with torch.no_grad():
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = generator(noise, labels.to(device))
    fake_uint8 = to_uint8(fake_images.cpu())
    real_uint8 = to_uint8(real_images.cpu())
    fid_metric.update(real_uint8.to(device), real=True)
    fid_metric.update(fake_uint8.to(device), real=False)
    return fid_metric.compute().item()


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

    for real_samples, real_labels in train_loader:
        for i in range(batch_size):
            ax = plt.subplot(math.ceil(math.sqrt(batch_size)), math.ceil(math.sqrt(batch_size)), i + 1)
            sample = denormalize(real_samples[i])
            plt.imshow(sample.permute(1, 2, 0))
            plt.xticks([]); plt.yticks([])
        break

    n_classes = 3
    generator = Generator(config['nz'], 64, 3, n_classes).to(device)
    discriminator = Discriminator(4, 64, n_classes).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)


    loss_function = nn.BCEWithLogitsLoss()
    optimizer_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    logger = Logger('gan-training-conditional').get_logger()


    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    fixed_noise = torch.randn(64, generator_input_size, 1, 1).to(device)

    for epoch in range(num_epochs + 1):
        start = time.time()
        for n, (real_samples, real_labels) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            real_labels = real_labels.to(device)
            b_size = real_samples.size(0)

            real_targets = torch.full((b_size,), 0.9, dtype=torch.float, device=device)
            fake_targets = torch.zeros(b_size, dtype=torch.float, device=device)

            discriminator.zero_grad()
            generator.zero_grad()

            class_losses_D = {}
            class_losses_G = {}

            unique_labels = torch.unique(real_labels)

            loss_discriminator = torch.tensor(0.0, device=device)
            loss_generator = torch.tensor(0.0, device=device)

            for label_class in unique_labels:
                class_mask = real_labels == label_class
                class_indices = class_mask.nonzero(as_tuple=True)[0]

                real_class = real_samples[class_indices]
                labels_class = real_labels[class_indices]

                real_targets_class = torch.full((real_class.size(0),), 0.9, dtype=torch.float, device=device)
                fake_targets_class = torch.zeros(real_class.size(0), dtype=torch.float, device=device)


                output_real = discriminator(real_class, labels_class).view(-1)
                loss_real = loss_function(output_real, real_targets_class)


                noise_class = torch.randn(real_class.size(0), generator_input_size, 1, 1, device=device)
                fake_class = generator(noise_class, labels_class)
                output_fake = discriminator(fake_class.detach(), labels_class).view(-1)
                loss_fake = loss_function(output_fake, fake_targets_class)

                loss_D = loss_real + loss_fake
                loss_D.backward(retain_graph=True)
                loss_discriminator += loss_D
                class_losses_D[int(label_class)] = loss_D.item()


                output_gen = discriminator(fake_class, labels_class).view(-1)
                loss_G = loss_function(output_gen, real_targets_class)
                loss_G.backward(retain_graph=True)
                loss_generator += loss_G
                class_losses_G[int(label_class)] = loss_G.item()

            optimizer_discriminator.step()


            optimizer_generator.step()

            if n % 50 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch {n}/{len(train_loader)} "
                      f"Loss D: {loss_discriminator.item():.4f}, Loss G: {loss_generator.item():.4f}")


        logger.log({
            'epoch': epoch,
            'loss_discriminator_total': loss_discriminator,
            'loss_generator_total': loss_generator,
            'loss_D_cat': class_losses_D.get(0, 0),
            'loss_D_dog': class_losses_D.get(1, 0),
            'loss_D_wild': class_losses_D.get(2, 0),
            'loss_G_cat': class_losses_G.get(0, 0),
            'loss_G_dog': class_losses_G.get(1, 0),
            'loss_G_wild': class_losses_G.get(2, 0)
        }, step=epoch)

        print("Time for epoch:", round(time.time() - start, 2), "seconds")

        if epoch % 5 == 0:
            generator.eval()
            with torch.no_grad():
                # Sample fixed noise for consistent output
                sample_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2]
                sample_labels = torch.tensor(sample_labels, dtype=torch.long, device=device)
                fake_images = generator(fixed_noise, sample_labels).detach().cpu()
                fake_images = (fake_images + 1) / 2
                vutils.save_image(fake_images, f"{output_dir}/epoch_{epoch}.png", nrow=8, normalize=True)
                logger.log({
                    "DCGAN Generated Images": wandb.Image(
                        vutils.make_grid(fake_images, nrow=8).permute(1, 2, 0).numpy(), caption=f"Epoch: {epoch}")
                }, step=epoch)
            generator.train()
            real_batch, real_labels = next(iter(train_loader))
            real_batch = real_batch[:64].to(device)
            real_labels = real_labels[:64].to(device)

            fid = compute_fid_score(generator, real_batch, real_labels, device, config['nz'])
            logger.log({'epoch': epoch, "FID Score": fid}, step=epoch)
            print(f"FID Score at epoch {epoch}: {fid:.4f}")
            print("saving and logging images...")

    torch.save(generator.state_dict(), "animalImageGAN.pt")
    torch.save(generator, "animalImageGAN_full.pt")