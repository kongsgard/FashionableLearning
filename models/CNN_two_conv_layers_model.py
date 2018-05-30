from base.base_model import BaseModel
import tensorflow as tf


class CNNTwoConvLayersModel(BaseModel):
    def __init__(self, config):
        super(CNNTwoConvLayersModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # Create the model
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image")
        self.x_reshaped = tf.reshape(self.x, [-1, 28, 28, 1])

        # First convolutional layer
        self.w1 = self.weight_variable([5, 5, 1, 32])
        self.b1 = self.bias_variable([32])
        self.conv1 = self.conv2d(self.x_reshaped, self.w1) + self.b1
        self.act1 = tf.nn.relu(self.conv1)
        self.pool1 = self.max_pool_2x2(self.act1)

        # Second convolutional layer
        self.w2 = self.weight_variable([5, 5, 32, 64])
        self.b2 = self.bias_variable([64])
        self.conv2 = self.conv2d(self.pool1, self.w2) + self.b2
        self.act2 = tf.nn.relu(self.conv2)
        self.pool2 = self.max_pool_2x2(self.act2)

        # Flatten layer
        self.flatten = tf.reshape(self.pool2, [-1, 7 * 7 * 64])

        # Fully connected layer
        self.w3 = self.weight_variable([7 * 7 * 64, 1024])
        self.b3 = self.bias_variable([1024])
        self.fc3 = tf.matmul(self.flatten, self.w3) + self.b3
        self.act3 = tf.nn.relu(self.fc3)
        # Dropout, to avoid over-fitting
        self.keep_prob = tf.placeholder(tf.float32)
        self.drop3 = tf.nn.dropout(self.act3, self.keep_prob)

        # Readout layer
        self.w4 = self.weight_variable([1024, 10])
        self.b4 = self.bias_variable([10])
        self.logits = tf.matmul(self.drop3, self.w4) + self.b4

        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="label")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,logits=self.logits))
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cross_entropy,
                                                                                     global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    @staticmethod
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    def conv2d(features, weight):
        return tf.nn.conv2d(features, weight, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(features):
        return tf.nn.max_pool(features, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')