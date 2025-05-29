import os
import torch
from torch import optim
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
import time
import math
import multiprocessing
import random
import wandb
from dataset import get_data
from models import Discriminator, Generator, current_epoch
import models
from logger import Logger
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import KernelInceptionDistance
import torch.nn.functional as F
import torchvision.utils as vutils

# FID
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


def compute_gradient_penalty(D, real_samples, fake_samples, labels, alpha):
    if real_samples.size(2) != fake_samples.size(2) or real_samples.size(3) != fake_samples.size(3):
        fake_samples = F.interpolate(fake_samples, size=(real_samples.size(2), real_samples.size(3)), mode='bilinear',
                                     align_corners=False)

    alpha_rand = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = alpha_rand * real_samples + ((1 - alpha_rand) * fake_samples)
    interpolates.requires_grad_(True)

    # Pass `alpha` into the discriminator forward call
    d_interpolates = D(interpolates, labels, alpha)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def d_loss(real_scores, fake_scores, gp, λ=1):
    return -(torch.mean(real_scores) - torch.mean(fake_scores)) + λ * gp


def g_loss(fake_scores):
    return -torch.mean(fake_scores)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def add_instance_noise(images, stddev):
    if stddev > 0:
        noise = torch.randn_like(images) * stddev
        return images + noise
    return images


def get_noise_std(epoch, max_std=0.1, decay_epochs=50):
    return max_std * (1 - epoch / decay_epochs) if epoch < decay_epochs else 0.0


def adjust_resolution(epoch, initial_resolution=64, milestone_epochs=[1500,500,0], resolutions=[256,128,64]):
    for milestone, resolution in zip(milestone_epochs, resolutions):
        if epoch >= milestone:
            return resolution
    return initial_resolution

def compute_kid_score(generator, real_images, labels, device, nz, subset_size1=128):
    kid_metric = KernelInceptionDistance(subset_size=subset_size1).to(device)
    generator.eval()
    with torch.no_grad():
        batch_size = real_images.size(0)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = generator(noise, labels.to(device))

    fake_images_uint8 = to_uint8(fake_images.cpu())
    real_images_uint8 = to_uint8(real_images.cpu())

    kid_metric.update(real_images_uint8.to(device), real=True)
    kid_metric.update(fake_images_uint8.to(device), real=False)
    kid_result = kid_metric.compute()
    print(kid_result)
    return kid_result[0].item()
if __name__ == '__main__':
    multiprocessing.freeze_support()

    manualSeed = 42
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    training_set = get_data()
    batch_size = config['batch_size']
    generator_input_size = config['nz']
    num_epochs = config['epochs']
    learning_rate_gen = config['lr_g']
    learning_rate_dis = config['lr_d']
    beta1 = config["beta1"]
    beta2 = config["beta2"]

    sample_size = len(training_set) // batch_size * batch_size
    indices_list = list(range(sample_size))
    training_set = torch.utils.data.Subset(training_set, indices_list)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    n_classes = 3

    # Initial resolution
    current_resolution = 64

    generator = Generator(config['nz'], config['ngf'], config['nc'], 3, current_resolution).to(device)
    discriminator = Discriminator(config['nc'], config['ndf'], 3, current_resolution).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_gen, betas=(beta1, beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_dis, betas=(beta1, beta2))

    logger = Logger('PROSACDCGAN-NO-DATASETRESOLUTIONINCREASE').get_logger()
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    fixed_noise = torch.randn(64, generator_input_size, 1, 1).to(device)

    max_grad_norm = 5.0
    for epoch in range(num_epochs + 1):
        start = time.time()
        models.current_epoch = epoch
        alpha = 1.0
        if 500 <= epoch < 800:
            alpha = (1 - math.cos((epoch - 500) / 300 * math.pi)) / 2
            alpha = min(alpha, 1.0)
        if 1500 <= epoch < 2000:
            alpha = (1 - math.cos((epoch - 1500) / 500 * math.pi)) / 2
            alpha = min(alpha, 1.0)
        print(epoch)
        new_resolution = adjust_resolution(epoch)
        print("new resolution output: ", new_resolution)
        if new_resolution > current_resolution:
            print(f"Resolution changing to {new_resolution} at epoch {epoch}")
            new_generator = Generator(config['nz'], config['ngf'], config['nc'], 3, new_resolution).to(device)
            new_discriminator = Discriminator(config['nc'], config['ndf'], 3, new_resolution).to(device)
            generator.resolution = new_resolution
            new_generator.load_state_dict(generator.state_dict(), strict=False)
            new_discriminator.load_state_dict(discriminator.state_dict(), strict=False)
            generator = new_generator
            discriminator = new_discriminator
            optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_gen, betas=(beta1, beta2))
            optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_dis, betas=(beta1, beta2))
            current_resolution = new_resolution
            #Training_set = get_data(None,current_resolution)
            #sample_size = len(training_set) // batch_size * batch_size
            #indices_list = list(range(sample_size))
            #training_set = torch.utils.data.Subset(training_set, indices_list)
            #train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=8)
    
        noise_std = get_noise_std(epoch)
        loss_discriminator = 0.0
        loss_generator = 0.0
        class_losses_D = {}
        class_losses_G = {}
    
        for n, (real_images, labels) in enumerate(train_loader):
            real_images = real_images.to(device)
            labels = labels.to(device)
            b_size = real_images.size(0)
    
            if current_resolution == 128 and real_images.size(2) == 64:
                real_images = F.interpolate(real_images, size=128)
    
            unique_labels = torch.unique(labels)
    
            for label_class in unique_labels:
                class_mask = labels == label_class
                class_indices = class_mask.nonzero(as_tuple=True)[0]
                real_class = real_images[class_indices]
                labels_class = labels[class_indices]
    
                noise_class = torch.randn(real_class.size(0), config['nz'], 1, 1, device=device)
                fake_class = generator(noise_class, labels_class, alpha)
                noisy_real = add_instance_noise(real_class, noise_std)
                noisy_fake = add_instance_noise(fake_class.detach(), noise_std)
                discriminator.zero_grad()
                real_scores = discriminator(noisy_real, labels_class, alpha).view(-1)
                fake_scores = discriminator(noisy_fake, labels_class, alpha).view(-1)
    
                gp = compute_gradient_penalty(discriminator, real_class.data, fake_class.data, labels_class, alpha)
                loss_D = d_loss(real_scores, fake_scores, gp)
                loss_D.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_grad_norm)
                optimizer_D.step()
                loss_discriminator += loss_D.item()
                class_losses_D[int(label_class)] = loss_D.item()
                mean_real_score = torch.mean(real_scores).item()
                mean_fake_score = torch.mean(fake_scores).item()
                # Train Generator
                generator.zero_grad()
                fake_scores_G = discriminator(fake_class, labels_class, alpha).view(-1)
                loss_G = g_loss(fake_scores_G)
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
                optimizer_G.step()
                loss_generator += loss_G.item()
                class_losses_G[int(label_class)] = loss_G.item()
    
        # Logging after each epoch
        logger.log({
            'epoch': epoch,
            'loss_discriminator_total': loss_discriminator,
            'loss_generator_total': loss_generator,
            'loss_D_cat': class_losses_D.get(0, 0),
            'loss_D_dog': class_losses_D.get(1, 0),
            'loss_D_wild': class_losses_D.get(2, 0),
            'loss_G_cat': class_losses_G.get(0, 0),
            'loss_G_dog': class_losses_G.get(1, 0),
            'loss_G_wild': class_losses_G.get(2, 0),
            'instance_noise_std': noise_std,
            'Discriminator_Mean_real': mean_real_score,
            'Discriminator_mean_fake': mean_fake_score
        }, step=epoch)
    
        if epoch % 5 == 0:
            generator.eval()
            with torch.no_grad():
                num_samples = fixed_noise.size(0)
                sample_labels = torch.tensor(
                    ([0] * (num_samples // 3)) +
                    ([1] * (num_samples // 3)) +
                    ([2] * (num_samples - 2 * (num_samples // 3))),
                    dtype=torch.long, device=device
                )
    
                fake_images = generator(fixed_noise, sample_labels, alpha).detach().cpu()
                print(fake_images.shape)
                fake_images = (fake_images + 1) / 2
                vutils.save_image(fake_images, f"{output_dir}/epoch_{epoch}.png", nrow=8, normalize=True)
    
                logger.log({
                    "Generated Images": wandb.Image(
                        vutils.make_grid(fake_images, nrow=8).permute(1, 2, 0).numpy(),
                        caption=f"Epoch: {epoch}")
                }, step=epoch)
    
            generator.train()  # Switch back to training mode
            real_batch, real_labels = next(iter(train_loader))
            real_batch = real_batch[:64].to(device)
            real_labels = real_labels[:64].to(device)
    
            # Compute FID and KID scores
            fid = compute_fid_score(generator, real_batch, real_labels, device, config['nz'])
            kid = compute_kid_score(generator, real_batch, real_labels, device, config['nz'], real_batch.size(0))
            torch.save(generator.state_dict(), "animalImageGAN.pt")
            torch.save(generator, "animalImageGAN_full.pt")
            logger.log({'epoch': epoch, "KID Score": kid}, step=epoch)
            logger.log({'epoch': epoch, "FID Score": fid}, step=epoch)
    
    torch.save(generator.state_dict(), "animalImageGAN.pt")
    torch.save(generator, "animalImageGAN_full.pt")

