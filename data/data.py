import random
import logging
import os
import json

def load_set(folder_path, only_dict=False):
    """
    Load Training Set

    Function that takes Dataset Directory,
    finds image classes and returns list of filenames
    and their labels

        Args:
        - folder_path: path to Dataset folder
        - only-dict: Boolean variable if only classes dictionary is needed
        Default false

        Returns:
        - classes_dict: dictionary of index : 'class'
        - filenames: list of all dataset images
        - labels: list of images classes
    """
    # Config Logging
    logging.basicConfig(level=logging.INFO)

    # Get Classes in Dataset Folder
    files = os.listdir(folder_path)
    classes = list(filter(lambda file: os.path.isdir(os.path.join(folder_path, file)), files))
    n_classes = len(classes)

    # Log Classes to Command Line
    logging.info("---------------------------------".format(n_classes))
    logging.info("               Found {0} classes".format(n_classes))
    logging.info("    Listing them below in format index - class")

    # Create Classes Dict
    classes_dict = {}
    for i, img_class in enumerate(classes):
        logging.info("          index    {}:      {}".format(i, img_class))
        classes_dict[i] = img_class

    # If Only Description for each index is required (In Prediction Script)
    if only_dict:
        return classes_dict

    # Define function for generating label
    def generate_label(index, n_classes):
        label = [0.0] * n_classes
        label[index] = 1.0
        return label

    # Get filnames and labels list
    filenames = []
    labels = []

    for i, img_class in enumerate(classes):
        # count number of images in each class
        n_class_img = 0
        img_class_path = os.path.join(folder_path, img_class)
        images = os.listdir(img_class_path)
        for img in images:
            if img.endswith('.png') or img.endswith('.jpg'):
                img_path = os.path.join(img_class_path, img)
                filenames.append(img_path)
                labels.append(generate_label(i, n_classes))
                n_class_img += 1

        # log number of images in class
        logging.info("     Found {} images for {} class".format(n_class_img, img_class))

    return classes_dict, filenames, labels


def divide_set(filenames, labels):
    """
    Divide Training set

    Function that takes training set
    and randomly divides it in two parts:
    training set  -- 80%
    validation set -- 20%

        Args:
        - filnames: array of paths to training images
        - labels: array of class-probabilities for each training image

        Returns:
        - train_filenames: 80% of images
        - train_labels: probabilities of the training images
        - valid_filenames: 20% of images
        - valid_labels: probabilities of the validation images
    """
    # Config Logging
    logging.basicConfig(level=logging.INFO)

    # Get number of images
    dataset_size = len(filenames)

    # Create dictionary filename: it's scores and shuffle filenames
    dictionary = dict(zip(filenames, labels))
    random.shuffle(filenames)

    train_filenames = []
    train_labels = []

    valid_filenames = []
    valid_labels = []

    # Calculate 80% and 20% of number of images
    train_dataset_size = dataset_size * 8 // 10
    valid_dataset_size = dataset_size - train_dataset_size

    # Log number of train and valid samples
    logging.info("-----------------------------------------")
    logging.info("   Found {0} images for training dataset".format(train_dataset_size))
    logging.info("   Found {0} images for validation dataset".format(valid_dataset_size))
    logging.info("-----------------------------------------")

    # Split samples in Train and Val sets
    for i, file in enumerate(filenames):
        if i <= train_dataset_size:
            train_filenames.append(file)
            train_labels.append(dictionary[file])
        else:
            valid_filenames.append(file)
            valid_labels.append(dictionary[file])

    return train_filenames, train_labels, valid_filenames, valid_labels


def parse_json(configfile) -> tuple:
    with open(configfile) as jsonfile:
        dict = json.load(jsonfile)

        try:
            image = dict['image']
            optimizer = dict['optimizer']
            training = dict['training']
        except:
            print("Failed parsing JSON: NO 'Image', 'Optimizer', 'Training'")
            exit()

        try:
            image_size = image['image_size']
            n_channels = image['n_channels']
        except:
            print("Failed parsing JSON: NO 'image_size', 'n_channels' inside 'image'")
            exit()

        try:
            lr = optimizer['lr']
            beta1 = optimizer['beta1']
            beta2 = optimizer['beta2']
            epsilon = optimizer['epsilon']
        except:
            print("Failed parsing JSON: NO 'lr', 'beta1', 'beta2', 'epsilon' inside 'optimizer'")
            exit()

        try:
            batch_size = training['batch_size']
        except:
            print("Failed parsing JSON: NO 'batch_size' inside 'training'")
            exit()

        return (image_size, n_channels), (lr, beta1, beta2, epsilon), batch_size


def get_images_from_folder(folder) -> list:
    filenames = []

    files = os.listdir(folder)
    for file in files:
        if file.endswith('.png') or file.endswith('.jpg'):
            img_path = os.path.join(folder, file)
            filenames.append(img_path)

    return filenames
