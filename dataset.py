import pandas as pd
import os
import torchvision.transforms as transforms
import torchvision
import yaml

def get_data():

    cwd = os.getcwd()

    #we decided to combine val and train for more training images as validation set is not required for dcgans

    # train_file_names = []
    # for root, dirs, filenames in os.walk(cwd + '/afhq/train'):
    #     for filename in filenames:
    #         train_file_names.append(filename)
    # print("Number of Training Data: " +str(len(train_file_names)))
    #
    # #length of val images
    # val_file_names = []
    # for root, dirs, filenames in os.walk(cwd + '/afhq/val'):
    #     for filename in filenames:
    #         val_file_names.append(filename)
    # print("Number of Validation Data: " +str(len(val_file_names)))
    #
    # #confirms no overlaps in files, val images are unique to train images
    # matches = list(set(train_file_names) & set(val_file_names))
    # print(len(matches))

    train_file_names = []
    for root, dirs, filenames in os.walk(cwd + '/dataAnimals/train'):
        for filename in filenames:
            train_file_names.append(filename)
    print("Total training data: " +str(len(train_file_names)))

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    image_size =  config['image_size']
    root_path = cwd + "/dataAnimals/train/"

    #makes all images 256x256 and transforms them to a tensor
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()  # No normalization here
    ])

    training_set = torchvision.datasets.ImageFolder(root=root_path, transform=transform)

    return training_set

#training_set = get_data()
#print(training_set)