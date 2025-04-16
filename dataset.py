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
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    training_set = torchvision.datasets.ImageFolder(root=root_path, transform=transform)

    #counts how many of each label appear, takes a while to run so itll be left commented out.
    # cats=0 #5653
    # dogs=0 #5239
    # wild=0 #5238
    # for i in range(len(training_set)):
    #     image, label = training_set[i]
    #     if label == 0:
    #         cats += 1
    #     elif label == 1:
    #         dogs += 1
    #     else:
    #         wild += 1
    #
    # print("Training image distribution: Cats = "+str(cats)+", Dogs = "+str(dogs)+", Wild = "+str(wild))


    return training_set


#train_set = get_data()
