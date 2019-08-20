import tensorflow as tf
from model.resnet import ResNet50
from data.data import load_set

dataset_path = ''

if __name__ == '__main__':
    with tf.Session() as sess:
        model = ResNet50()

        classes_dict, filenames, labels = load_set(dataset_path)

        model.load_data(filenames, labels)

        model.build()
        model.train(500)
