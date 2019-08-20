import random
import logging
import os

def load_set(folder_path):
    """
    Load Training Set

    Function that takes Dataset Directory,
    finds image classes and returns list of filenames
    and their labels

        Args:
        - folder_path: path to Dataset folder

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

    logging.info("               Found {0} classes".format(n_classes))
    logging.info("    Listing them below in format index - class")

    classes_dict = {}
    for i, img_class in enumerate(classes):
        logging.info("          index    {}:      {}".format(i, img_class))
        classes_dict[i] = img_class

    filenames = []
    labels = []

    for i, img_class in enumerate(classes):
        img_class_path = os.path.join(folder_path, img_class)
        images = os.listdir(img_class_path)
        for img in images:
            img_path = os.path.join(img_class_path, img)
            filenames.append(img_path)
            labels.append(generate_label(i, n_classes))

def generate_label(index, n_classes):
    label = [0.0] * n_classes
    label[index] = 1.0
    return label

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
    dataset_size = len(filenames)

    dictionary = dict(zip(filenames, labels))

    random.shuffle(filenames)

    train_filenames = []
    train_labels = []

    valid_filenames = []
    valid_labels = []

    train_dataset_size = dataset_size * 8 // 10
    valid_dataset_size = dataset_size - train_dataset_size

    print("-----------------------------------------")
    print("Found {0} images for training dataset".format(train_dataset_size))
    print("Found {0} images for validation dataset".format(valid_dataset_size))
    print("-----------------------------------------")

    for i, file in enumerate(filenames):
        if i <= train_dataset_size:
            train_filenames.append(file)
            train_labels.append(dictionary[file])
        else:
            valid_filenames.append(file)
            valid_labels.append(dictionary[file])

    return train_filenames, train_labels, valid_filenames, valid_labels
