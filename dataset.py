import pandas as pd
import os
import torchvision.transforms as transforms
import torchvision
import yaml
from torch.utils.data import Dataset
from PIL import Image


def get_data(transform=None, image_resolution=64):
    cwd = os.getcwd()
    # Count training samples
    train_file_names = []
    for root, dirs, filenames in os.walk(cwd + '/dataAnimals/train'):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                train_file_names.append(filename)
    print(f"Total training data: {len(train_file_names)}")

    # Load configuration file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract image_size from config, or default to the given image_resolution argument
    image_size = image_resolution  # Using the dynamic resolution passed in
    root_path = os.path.join(cwd, "dataAnimals/train/")

    # Enhanced transforms with augmentation
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.125)),  # Resize to slightly larger than target resolution
            transforms.RandomCrop(image_size),  # Crop to the target resolution
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])

    # Create the dataset with the applied transformations
    training_set = torchvision.datasets.ImageFolder(
        root=root_path,
        transform=transform,
        is_valid_file=lambda x: x.lower().endswith(('.png', '.jpg', '.jpeg'))
    )

    # Print class distribution
    class_counts = pd.Series([label for _, label in training_set]).value_counts()
    print("\nClass distribution:")
    print(class_counts.to_string())

    return training_set