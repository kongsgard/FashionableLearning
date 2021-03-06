from base.base_model import BaseModel
from tensorflow.contrib.layers import xavier_initializer
import tensorflow as tf
import numpy as np

class VGG16Model(BaseModel):
    def __init__(self, config):
        super(VGG16Model, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # Create the model
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='x')
        self.x_reshaped = tf.reshape(self.x, [-1, 28, 28, 1], name='Reshape_input')

        # Block 1 -- outputs 112x112x64
        with tf.name_scope("Block_1"):
            self.conv1_1 = tf.layers.conv2d(inputs=self.x_reshaped,filters = 64,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.conv1_2 = tf.layers.conv2d(inputs=self.conv1_1,filters = 64,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1_2, pool_size=[2, 2], strides=2, padding = 'same')

        # Block 2 -- outputs 56x56x128
        with tf.name_scope("Block_2"):
            self.conv2_1 = tf.layers.conv2d(inputs=self.pool1,filters = 128,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.conv2_2 = tf.layers.conv2d(inputs=self.conv2_1,filters = 128,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2_2, pool_size=[2, 2], strides=2, padding = 'same')

        # Block 3 -- outputs 28x28x256
        with tf.name_scope("Block_3"):
            self.conv3_1 = tf.layers.conv2d(inputs=self.pool2,filters = 256,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.conv3_2 = tf.layers.conv2d(inputs=self.conv3_1,filters = 256,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.conv3_3 = tf.layers.conv2d(inputs=self.conv3_2,filters = 256,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.pool3 = tf.layers.max_pooling2d(inputs=self.conv3_3, pool_size=[2, 2], strides=2, padding = 'same')

        # Block 4 -- outputs 14x14x512
        with tf.name_scope("Block_4"):
            self.conv4_1 = tf.layers.conv2d(inputs=self.pool3,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.conv4_2 = tf.layers.conv2d(inputs=self.conv4_1,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.conv4_3 = tf.layers.conv2d(inputs=self.conv4_2,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.pool4 = tf.layers.max_pooling2d(inputs=self.conv4_3, pool_size=[2, 2], strides=2, padding = 'same')

        # Block 5 -- outputs 7x7x512
        with tf.name_scope("Block_5"):
            self.conv5_1 = tf.layers.conv2d(inputs=self.pool4,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.conv5_2 = tf.layers.conv2d(inputs=self.conv5_1,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.conv5_3 = tf.layers.conv2d(inputs=self.conv5_2,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
            self.pool5 = tf.layers.max_pooling2d(inputs=self.conv5_3, pool_size=[2, 2], strides=2, padding = 'same')

        # Flatten layer
        flattened_shape = np.prod([s.value for s in self.pool5.get_shape()[1:]])
        self.flatten = tf.reshape(self.pool5, [-1, flattened_shape], name="flatten")

        #Fully connected layers
        with tf.name_scope("FC_layers"):
        self.fc6_1 = tf.contrib.layers.fully_connected(inputs=self.flatten, num_outputs=4096)
        self.fc6_2 = tf.contrib.layers.fully_connected(inputs=self.fc6_1, num_outputs=4096)
        self.fc6_3 = tf.contrib.layers.fully_connected(inputs=self.fc6_2, num_outputs=4096)
        self.logits = tf.contrib.layers.fully_connected(inputs=self.fc6_3, num_outputs=10,activation_fn=None)

        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='actual_label')

        with tf.name_scope("Loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

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
