import argparse
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model.resnet import ResNet50
from data.data import load_set

import tensorflow as tf
import numpy as np
import logging

from matplotlib import pyplot as plt
from PIL import Image

JSON_CONFIG = 'config.json'


def show_results(filename, classname, accuracy):
    image = Image.open(filename)
    plt.imshow(image)
    plt.title("This is {} with {}%".format(classname, accuracy))
    plt.show()


def interpret(filenames, predictions, classes_dict):
    assert len(filenames) == predictions.shape[0]

    for i, file in enumerate(filenames):
        print(file)
        prediction = predictions[i]
        class_index = np.argmax(prediction)
        accuracy = prediction[class_index]
        class_name = classes_dict[class_index]
        print("is {} with {}%".format(class_name, accuracy * 100))
        show_results(file, class_name, accuracy * 100)


def predict(model_folder, image_folder, classes_dict, debug=False):
    weights = os.path.join(model_folder, 'model.ckpt')
    n_classes = len(classes_dict)
    model = ResNet50(JSON_CONFIG, n_classes)
    filenames = model.load_pred(image_folder)
    predictions = model.predict(weights, debug=debug)
    interpret(filenames, predictions, classes_dict)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-img", "--img-folder", required=True,
                        help="specify path to images to make prediction")
    parser.add_argument("-f", "--data-folder", required=True,
                        help="path to Training Dataset to get class dict")
    parser.add_argument("-mod","--model-folder", required=True,
                        help="specify path to folder with saved model")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Use TensorFlow Debugger")

    args = parser.parse_args()

    model_folder = args.model_folder
    image_folder = args.img_folder
    dataset_folder = args.data_folder
    debug = args.debug

    classes_dict = load_set(dataset_folder, only_dict=True)
    logging.info(classes_dict)

    predict(model_folder, image_folder, classes_dict, debug)


if __name__ == '__main__':
    main()
