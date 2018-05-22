from base.base_model import BaseModel
import tensorflow as tf


class SimpleCNNModel(BaseModel):
    def __init__(self, config):
        super(SimpleCNNModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # TODO: Do not hardcode these variables, but acquire them from the fashion_mnist data
        num_features = 784
        num_classes = 10

        # Build the graph
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name="image")
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="label")

        # First convolutional layer
        first_conv_weight = self.weight_variable([5, 5, 1, 32])
        first_conv_bias = self.bias_variable([32])

        input_image = tf.reshape(self.x, [-1, 28, 28, 1])

        first_conv_activation = tf.nn.relu(self.conv2d(input_image, first_conv_weight) + first_conv_bias)
        first_conv_pool = self.max_pool_2x2(first_conv_activation)

        # Second convolutional layer
        second_conv_weight = self.weight_variable([5, 5, 32, 64])
        second_conv_bias = self.bias_variable([64])

        second_conv_activation = tf.nn.relu(self.conv2d(first_conv_pool, second_conv_weight) + second_conv_bias)
        second_conv_pool = self.max_pool_2x2(second_conv_activation)

        # Fully-connected layer (Dense Layer)
        dense_layer_weight = self.weight_variable([7 * 7 * 64, 1024])
        dense_layer_bias = self.bias_variable([1024])

        second_conv_pool_flatten = tf.reshape(second_conv_pool, [-1, 7 * 7 * 64])
        dense_layer_activation = tf.nn.relu(tf.matmul(second_conv_pool_flatten, dense_layer_weight) +
                                            dense_layer_bias)

        # Dropout, to avoid over-fitting
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(dense_layer_activation, self.keep_prob)

        # Readout layer
        readout_weight = self.weight_variable([1024, num_classes])
        readout_bias = self.bias_variable([num_classes])

        logits = tf.matmul(h_fc1_drop, readout_weight) + readout_bias
        with tf.name_scope('softmax'):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.y))

            self.train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cross_entropy,
                                                                                     global_step=self.global_step_tensor)

            with tf.name_scope('accuracy'):
                logits = tf.identity(tf.nn.softmax(logits), name='prediction')
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    @staticmethod
    def weight_variable(shape):
        """Returns a weight matrix consisting of arbitrary values.
        :param shape: The shape of the weight matrix to create.
        :return: The weight matrix consisting of arbitrary values.
        """
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """Returns a bias matrix consisting of 0.1 values.
        :param shape: The shape of the bias matrix to create.
        :return: The bias matrix consisting of 0.1 values.
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(features, weight):
        """Produces a convolutional layer that filters an image subregion
        :param features: The layer input.
        :param weight: The size of the layer filter.
        :return: Returns a convolutional layer.
        """
        return tf.nn.conv2d(features, weight, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(features):
        """Downsamples the image based on convolutional layer
        :param features: The input to downsample.
        :return: Downsampled input.
        """
        return tf.nn.max_pool(features, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
