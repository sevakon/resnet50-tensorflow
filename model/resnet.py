import tensorflow as tf
from model.layers import *
from data.utils import _parse_function, _parse_image, train_preprocess
from data.data import divide_set, get_images_from_folder, parse_json

from tensorflow.python import debug as tf_debug

import logging
import time


class ResNet50(object):
    '''
    ResNet 50 Model
    '''

    def __init__(self, config, n_classes):
        # Config Logging and log TF Version
        logging.basicConfig(level=logging.INFO)
        logging.info("-----------------------------------------")
        logging.info("         USING TF Version {}".format(tf.__version__))

        self.is_training = True

        # load hyperparameters with config file
        # (image), (optimizer), batch_size
        (image_size, n_channels), (lr, beta1, beta2, epsilon), batch_size = parse_json(config)

        # image
        self.image_size = image_size
        self.n_channels = n_channels

        # optimizer
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # batch size
        self.batch_size = batch_size

        # Output Layer
        self.num_classes = n_classes

        # Input Layer
        self.input_tensor = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.n_channels])

        logging.info("---------------------------------------")
        logging.info("             MODEL CONFIG              ")
        logging.info("               OPTIMIZER               ")
        logging.info(" LR: {}, BETA1: {}, BETA2: {}, EPSILON: {}".format(self.lr, self.beta1, self.beta2, self.epsilon))
        logging.info("                 IMAGE                 ")
        logging.info("  IMAGE SIZE: {}, CHANNELS_NUMBER: {}  ".format(self.image_size, self.n_channels))
        logging.info("                TRAINING               ")
        logging.info("             BATCH SIZE: {}            ".format(self.batch_size))



    def load_data(self, filenames, labels):
        '''
        Creates TF Training and Validation Datasets
        '''

        # Check if each image is classified
        assert len(filenames) == len(labels)
        num_samples = len(filenames)

        # Divides Dataset in two parts: Training and Validation
        train_f, train_l, valid_f, valid_l = divide_set(filenames, labels)

        # Define lambda function for parsing images and augmentation
        parse_fn = lambda f, l: _parse_function(f, l, self.n_channels, self.image_size)
        train_fn = lambda f, l: train_preprocess(f, l)

        # Creates Iterator over the Training Set
        with tf.name_scope('train-data'):
            train_dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(train_f), tf.constant(train_l)))
                .shuffle(num_samples)
                .map(parse_fn, num_parallel_calls=4)
                .batch(self.batch_size)
                .prefetch(1)
            )

            train_iterator = train_dataset.make_initializable_iterator()
            self.train_iterator_init_op = train_iterator.initializer

        # Creates Iterator over the Validation Set
        with tf.name_scope('valid-data'):
            valid_dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(valid_f), tf.constant(valid_l)))
                .map(parse_fn, num_parallel_calls=4)
                .batch(self.batch_size)
                .prefetch(1)
            )

            valid_iterator = valid_dataset.make_initializable_iterator()
            self.valid_iterator_init_op = valid_iterator.initializer

        # Get Input and Output for Graph Inference
        self.images, self.labels = train_iterator.get_next()


    # ----------------------------  GRAPH  ----------------------------- #


    def inference(self):
        '''
        Defining model's graph
        '''
        # Stage 1
        self.conv1 = conv_layer(self.images, 7, self.n_channels, 64, 2, 'scale1')
        self.conv1 = bn(self.conv1, self.is_training, 'scale1')
        self.conv1 = relu(self.conv1)
        self.pool1 = maxpool(self.conv1, name='pool1')

        # Stage 2
        with tf.variable_scope('scale2'):
            self.block1_1 = res_block_3_layer(self.pool1, [64, 64, 256], 'block1_1', True, 1, self.is_training)
            self.block1_2 = res_block_3_layer(self.block1_1, [64, 64, 256], 'block1_2', False, 1, self.is_training)
            self.block1_3 = res_block_3_layer(self.block1_2, [64, 64, 256], 'block1_3', False, 1, self.is_training)

        # Stage 3
        with tf.variable_scope('scale3'):
            self.block2_1 = res_block_3_layer(self.block1_3, [128, 128, 512], 'block2_1', True, 2, self.is_training)
            self.block2_2 = res_block_3_layer(self.block2_1, [128, 128, 512], 'block2_2', False, 1, self.is_training)
            self.block2_3 = res_block_3_layer(self.block2_2, [128, 128, 512], 'block2_3', False, 1, self.is_training)
            self.block2_4 = res_block_3_layer(self.block2_3, [128, 128, 512], 'block2_4', False, 1, self.is_training)

        # Stage 4
        with tf.variable_scope('scale4'):
            self.block3_1 = res_block_3_layer(self.block2_4, [256, 256, 1024], 'block3_1', True, 2, self.is_training)
            self.block3_2 = res_block_3_layer(self.block3_1, [256, 256, 1024], 'block3_2', False, 1, self.is_training)
            self.block3_3 = res_block_3_layer(self.block3_2, [256, 256, 1024], 'block3_3', False, 1, self.is_training)
            self.block3_4 = res_block_3_layer(self.block3_3, [256, 256, 1024], 'block3_4', False, 1, self.is_training)
            self.block3_5 = res_block_3_layer(self.block3_4, [256, 256, 1024], 'block3_5', False, 1, self.is_training)
            self.block3_6 = res_block_3_layer(self.block3_5, [256, 256, 1024], 'block3_6', False, 1, self.is_training)

        # Stage 5
        with tf.variable_scope('scale5'):
            self.block4_1 = res_block_3_layer(self.block3_6, [512, 512, 2048], 'block4_1', True, 2, self.is_training)
            self.block4_2 = res_block_3_layer(self.block4_1, [512, 512, 2048], 'block4_2', False, 1, self.is_training)
            self.block4_3 = res_block_3_layer(self.block4_2, [512, 512, 2048], 'block4_3', False, 1, self.is_training)

        # Fully-Connected
        with tf.variable_scope('fc'):
            self.pool2 = avgpool(self.block4_3, 7, 1, 'pool2')
            self.logits = fc_layer(self.pool2, 2048, self.num_classes, 'fc1')


    # ---------------------------  TRAINING  ---------------------------- #


    def loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')


    def optimize(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            name='Adam'
        ).minimize(self.loss, global_step=self.global_step)


    def average_scalars(self):
        self.avg_loss = tf.Variable(0.0)
        self.avg_acc = tf.Variable(0.0)


    def eval(self):
        '''
        Count the number of right predictions in a batch
        '''
        with tf.name_scope('predict'):
            predictions = softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(predictions, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))


    def summary(self):
        '''
        Logging to TensorBoard
        '''
        with tf.name_scope('batch-summaries'):
            tf.summary.scalar('batch-loss', self.loss)
            tf.summary.scalar('batch-accuracy', self.accuracy)
            tf.summary.histogram('histogram-loss', self.loss)
            self.summary_op = tf.summary.merge_all()

        with tf.name_scope('trainig'):
            loss_summary = tf.summary.scalar('loss', self.avg_loss)
            acc_summary = tf.summary.scalar('acc', self.avg_acc)
            self.avg_summary_op = tf.summary.merge([loss_summary, acc_summary])


    def build(self):
        '''
        Build computational graph
        '''
        self.inference()
        self.loss()
        self.optimize()
        self.average_scalars()
        self.eval()
        self.summary()


    def write_average_summary(self, sess, writer, epoch, avg_loss, avg_acc):
        summaries = sess.run(self.avg_summary_op, {self.avg_loss: avg_loss, self.avg_acc: avg_acc})
        writer.add_summary(summaries, global_step=epoch)
        writer.flush()


    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        total_acc = 0
        n_batches = 0
        try:
            while True:
                _, loss_batch, acc_batch, summaries = sess.run([self.optimizer, self.loss, self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_loss += loss_batch
                total_acc += acc_batch
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        avg_loss = total_loss/n_batches
        avg_acc = total_acc/n_batches/self.batch_size
        self.write_average_summary(sess, writer, epoch, avg_loss, avg_acc)
        logging.info('Training loss at epoch {0}: {1}'.format(epoch, avg_loss))
        logging.info('Training accuracy at epoch {0}: {1}'.format(epoch, avg_acc))
        logging.info('Took: {0} seconds'.format(time.time() - start_time))
        return step + n_batches


    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_acc = 0
        total_loss = 0
        n_batches = 0
        try:
            while True:
                loss_batch, accuracy_batch, summaries = sess.run([self.loss, self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_loss += loss_batch
                total_acc += accuracy_batch
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass

        avg_loss = total_loss/n_batches
        avg_acc = total_acc/n_batches/self.batch_size
        self.write_average_summary(sess, writer, epoch, avg_loss, avg_acc)
        logging.info('Validation loss at epoch {0}: {1} '.format(epoch, avg_loss))
        logging.info('Validation accuracy at epoch {0}: {1} '.format(epoch, avg_acc))
        logging.info('Took: {0} seconds'.format(time.time() - start_time))
        return step + n_batches


    def train(self, n_epochs, debug=False):
        '''
        This train function alternates between training and evaluating once per epoch run
        '''
        # Config Logging
        logging.basicConfig(level=logging.INFO)

        train_writer = tf.summary.FileWriter('logs/train')
        val_writer = tf.summary.FileWriter('logs/val')

        train_writer.add_graph(tf.get_default_graph())

        with tf.Session() as sess:
            # Wrap Debug Session
            if debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=100)
            # upload existing saves
            train_step = self.global_step.eval()
            val_step = train_step
            for epoch in range(n_epochs):
                train_step = self.train_one_epoch(sess, saver, self.train_iterator_init_op, train_writer, epoch, train_step)
                val_step = self.eval_once(sess, self.train_iterator_init_op, val_writer, epoch, val_step)
                # Save Each Epoch
                save_path = saver.save(sess, "training/epoch{}/model.ckpt".format(epoch))
        writer.close()


    # ---------------------------  PREDICTION  ---------------------------- #


    def load_pred(self, folder):
        '''
        Creates Prediction Dataset
        '''
        # Get All Images From Folder
        filenames = get_images_from_folder(folder)
        # Calculating Batch Size
        batch = len(filenames)
        logging.info("       FOUND {0} IMAGES TO PREDICT".format(batch))

        # Parsing Images
        parse_fn = lambda f: _parse_image(f, self.n_channels, self.image_size)

        # Creates Iterator over the Prediction DataSet
        with tf.name_scope('predict-data'):
            predict_dataset = (tf.data.Dataset.from_tensor_slices(tf.constant(filenames))
                .map(parse_fn, num_parallel_calls=4)
                .batch(batch)
            )

            predict_iterator = predict_dataset.make_initializable_iterator()
            self.predict_iterator_init_op = predict_iterator.initializer

        # Pass Batch of Images into Model's Input
        self.images = predict_iterator.get_next()
        # Return filenames to match them with Predictions
        return filenames


    def predict(self, weights, is_logging=False, debug=False):
        '''
        Loads graph and weights,
        creates a feed dict and passes it through the model
        returns predictions
        '''
        # Build Computational Graph
        self.inference()
        self.training = False

        # Logging to TensorBoard
        if is_logging:
            writer = tf.summary.FileWriter('logs/predict')
            writer.add_graph(tf.get_default_graph())
            writer.close()

        # Get Predictions
        with tf.Session() as sess:
            # Wrap Debug Session
            if debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            # Restoring Weights From Trained Model
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, weights)
            logging.info("Finished restoring weights")

            # Passing images through Model
            start_time = time.time()
            logging.info("Started evaluating images")
            sess.run(self.predict_iterator_init_op)
            predictions = sess.run(softmax(self.logits))
            logging.info('Took: {0} seconds'.format(time.time() - start_time))

        # Return Prediction for further Results Interpretation
        return predictions
