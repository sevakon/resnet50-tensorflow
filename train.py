import argparse
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from model.resnet import ResNet50
from data.data import load_set

JSON_CONFIG = 'config.json'

def train(dir, n_epochs, debug=False):
    # Load Dataset
    classes_dict, filenames, labels = load_set(dir)
    n_classes = len(classes_dict)

    # Build model and load data into it
    model = ResNet50(JSON_CONFIG, n_classes)
    model.load_data(filenames, labels)
    model.build()
    model.train(n_epochs, debug=debug)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=sys.maxsize,
                        help="specify number of epochs for training, default is inf")
    parser.add_argument("-f", "--folder", type=str, required=True,
                        help="specify full path to dataset folder, no default")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Use TensorFlow Debugger")

    args = parser.parse_args()

    folder = args.folder
    epochs = args.epochs
    debug = args.debug
    train(folder, epochs, debug)


if __name__ == '__main__':
    main()
