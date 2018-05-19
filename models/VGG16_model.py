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
        self.conv1_1 = self.conv_layer(self.x_reshaped, name='block1_conv1', kh=3, kw=3, n_out=64)
        self.conv1_2 = self.conv_layer(self.conv1_1, name='block1_conv2', kh=3, kw=3, n_out=64)
        self.pool1 = self.max_pool(self.conv1_2,name='block1_pool', kh=2, kw=2, dw=2, dh=2)

        # Block 2 -- outputs 56x56x128
        self.conv2_1 = self.conv_layer(self.pool1,name='block2_conv1', kh=3, kw=3, n_out=128)
        self.conv2_2 = self.conv_layer(self.conv2_1,name='block2_conv2', kh=3, kw=3, n_out=128)
        self.pool2 = self.max_pool(self.conv2_2,name='block2_pool', kh=2, kw=2, dw=2, dh=2)

        # Block 3 -- outputs 28x28x256
        self.conv3_1 = self.conv_layer(self.pool2,name='block3_conv1', kh=3, kw=3, n_out=256)
        self.conv3_2 = self.conv_layer(self.conv3_1,name='block3_conv2', kh=3, kw=3, n_out=256)
        self.conv3_3 = self.conv_layer(self.conv3_2,name='block3_conv3', kh=3, kw=3, n_out=256)
        self.pool3 = self.max_pool(self.conv3_3,name='block3_pool', kh=2, kw=2, dw=2, dh=2)

        # Block 4 -- outputs 14x14x512
        self.conv4_1 = self.conv_layer(self.pool3,name='block4_conv1', kh=3, kw=3, n_out=512)
        self.conv4_2 = self.conv_layer(self.conv4_1,name='block4_conv2', kh=3, kw=3, n_out=512)
        self.conv4_3 = self.conv_layer(self.conv4_2,name='block4_conv3', kh=3, kw=3, n_out=512)
        self.pool4 = self.max_pool(self.conv4_3,name='block4_pool', kh=2, kw=2, dw=2, dh=2)

        # Block 5 -- outputs 7x7x512
        self.conv5_1 = self.conv_layer(self.pool4, name='block5_conv1', kh=3, kw=3, n_out=512)
        self.conv5_2 = self.conv_layer(self.conv5_1, name='block5_conv2', kh=3, kw=3, n_out=512)
        self.conv5_3 = self.conv_layer(self.conv5_2, name='block5_conv3', kh=3, kw=3, n_out=512)
        self.pool5 = self.max_pool(self.conv5_3, name='block5_pool', kh=2, kw=2, dw=2, dh=2)


        flattened_shape = np.prod([s.value for s in self.pool5.get_shape()[1:]])
        self.flatten = tf.reshape(self.pool5, [-1, flattened_shape], name="flatten")
        # Fully connected layer
        self.fc6_1 = self.fc_layer(self.flatten,name="block6_fc1", n_out=4096)
        self.fc6_2 = self.fc_layer(self.fc6_1,"block6_fc2", n_out=4096)
        self.fc6_3 = self.fc_layer(self.fc6_2,"block6_fc3", n_out=4096)
        self.out = self.fc_layer(self.fc6_3,"block6_out", n_out=10)

        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='actual_label')

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.out))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(self.out, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def fc_layer(self, input_tensor, name, n_out, activation_fn=tf.nn.relu):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weights = tf.get_variable('weights', [n_in, n_out], tf.float32, xavier_initializer())
            biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
            logits = tf.nn.bias_add(tf.matmul(input_tensor, weights), biases)
            return activation_fn(logits)

    def conv_layer(self, input_tensor, name, kw, kh, n_out, dw=1, dh=1, activation_fn=tf.nn.relu):
        n_in = input_tensor.get_shape()[-1].value
        with tf.variable_scope(name):
            weights = tf.get_variable('weights', [kh, kw, n_in, n_out], tf.float32, xavier_initializer())
            biases = tf.get_variable("bias", [n_out], tf.float32, tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(input_tensor, weights, (1, dh, dw, 1), padding='SAME')
            activation = activation_fn(tf.nn.bias_add(conv, biases))
            return activation

    def max_pool(self, input_tensor, name, kh, kw, dh, dw):
        return tf.nn.max_pool(input_tensor,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)
