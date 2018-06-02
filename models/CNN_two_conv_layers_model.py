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
        with tf.name_scope("Conv_layer1"):
            self.w1 = self.weight_variable([5, 5, 1, 32])
            self.b1 = self.bias_variable([32])
            self.conv1 = self.conv2d(self.x_reshaped, self.w1) + self.b1
            self.act1 = tf.nn.relu(self.conv1)
            self.batch_norm1 = self.batch_norm(self.act1, 32, self.is_training)
            self.pool1 = self.max_pool_2x2(self.batch_norm1)

        # Second convolutional layer
        with tf.name_scope("Conv_layer2"):
            self.w2 = self.weight_variable([5, 5, 32, 64])
            self.b2 = self.bias_variable([64])
            self.conv2 = self.conv2d(self.pool1, self.w2) + self.b2
            self.act2 = tf.nn.relu(self.conv2)
            self.batch_norm2 = self.batch_norm(self.act2, 64, self.is_training)
            self.pool2 = self.max_pool_2x2(self.batch_norm2)

        # Flatten layer
        self.flatten = tf.reshape(self.pool2, [-1, 7 * 7 * 64])

        # Fully connected layer
        with tf.name_scope("FC_layer"):
            self.w3 = self.weight_variable([7 * 7 * 64, 1024])
            self.b3 = self.bias_variable([1024])
            self.fc3 = tf.matmul(self.flatten, self.w3) + self.b3
            self.act3 = tf.nn.relu(self.fc3)

        # Dropout, to avoid over-fitting
        with tf.name_scope("Dropout_layer"):
            self.keep_prob = tf.placeholder(tf.float32)
            self.drop3 = tf.nn.dropout(self.act3, self.keep_prob)

        # Readout layer
        with tf.name_scope("Readout_layer"):
            self.w4 = self.weight_variable([1024, 10])
            self.b4 = self.bias_variable([10])
            self.logits = tf.matmul(self.drop3, self.w4) + self.b4

        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="label")

        with tf.name_scope("Loss"):
            regularizer = tf.nn.l2_loss(self.w1) + tf.nn.l2_loss(self.w2) + tf.nn.l2_loss(self.w3) + tf.nn.l2_loss(self.w4)

            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,logits=self.logits) + self.config.beta*regularizer)
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

    @staticmethod
    def batch_norm(x, n_out, phase_train):
        with tf.variable_scope('Batch_norm'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                         name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                          name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed
