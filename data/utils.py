import tensorflow as tf

def _parse_function(filename, label, n_channels, size):
    """
    Returns resized and normalized image and its label
    """
    resized_image = _parse_image(filename, n_channels, size)
    return resized_image, label


def _parse_image(filename, n_channels, size):
    """Obtain the image from the filename (for both training and validation).
    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    image_decoded = tf.image.decode_jpeg(image_string, channels=n_channels)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size])

    return resized_image


def train_preprocess(filename, label, random_flip=True):
    """
    Data Augmentation
    """
