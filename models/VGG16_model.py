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
        self.conv1_1 = tf.layers.conv2d(inputs=self.x_reshaped,filters = 64,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm1_1 = self.batch_norm(self.conv1_1, 64, self.is_training)
        self.conv1_2 = tf.layers.conv2d(inputs=self.batch_norm1_1,filters = 64,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm1_2 = self.batch_norm(self.conv1_2, 64, self.is_training)
        self.pool1 = tf.layers.max_pooling2d(inputs=self.batch_norm1_2, pool_size=[2, 2], strides=2, padding = 'same')

        # Block 2 -- outputs 56x56x128
        self.conv2_1 = tf.layers.conv2d(inputs=self.pool1,filters = 128,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm2_1 = self.batch_norm(self.conv2_1, 128, self.is_training)
        self.conv2_2 = tf.layers.conv2d(inputs=self.batch_norm2_1,filters = 128,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm2_2 = self.batch_norm(self.conv2_2, 128, self.is_training)
        self.pool2 = tf.layers.max_pooling2d(inputs=self.batch_norm2_2, pool_size=[2, 2], strides=2, padding = 'same')

        # Block 3 -- outputs 28x28x256
        self.conv3_1 = tf.layers.conv2d(inputs=self.pool2,filters = 256,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm3_1 = self.batch_norm(self.conv3_1, 256, self.is_training)
        self.conv3_2 = tf.layers.conv2d(inputs=self.batch_norm3_1,filters = 256,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm3_2 = self.batch_norm(self.conv3_2, 256, self.is_training)
        self.conv3_3 = tf.layers.conv2d(inputs=self.batch_norm3_2,filters = 256,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm3_3 = self.batch_norm(self.conv3_3, 256, self.is_training)
        self.pool3 = tf.layers.max_pooling2d(inputs=self.batch_norm3_3, pool_size=[2, 2], strides=2, padding = 'same')

        # Block 4 -- outputs 14x14x512
        self.conv4_1 = tf.layers.conv2d(inputs=self.pool3,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm4_1 = self.batch_norm(self.conv4_1, 512, self.is_training)
        self.conv4_2 = tf.layers.conv2d(inputs=self.batch_norm4_1,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm4_2 = self.batch_norm(self.conv4_2, 512, self.is_training)
        self.conv4_3 = tf.layers.conv2d(inputs=self.batch_norm4_2,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm4_3 = self.batch_norm(self.conv4_3, 512, self.is_training)
        self.pool4 = tf.layers.max_pooling2d(inputs=self.batch_norm4_3, pool_size=[2, 2], strides=2, padding = 'same')

        # Block 5 -- outputs 7x7x512
        self.conv5_1 = tf.layers.conv2d(inputs=self.pool4,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm5_1 = self.batch_norm(self.conv5_1, 512, self.is_training)
        self.conv5_2 = tf.layers.conv2d(inputs=self.batch_norm5_1,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm5_2 = self.batch_norm(self.conv5_2, 512, self.is_training)
        self.conv5_3 = tf.layers.conv2d(inputs=self.batch_norm5_2,filters = 512,kernel_size = [3,3], padding='same',activation=tf.nn.relu)
        self.batch_norm5_3 = self.batch_norm(self.conv5_3, 512, self.is_training)
        self.pool5 = tf.layers.max_pooling2d(inputs=self.batch_norm5_3, pool_size=[2, 2], strides=2, padding = 'same')

        # Flatten layer
        flattened_shape = np.prod([s.value for s in self.pool5.get_shape()[1:]])
        self.flatten = tf.reshape(self.pool5, [-1, flattened_shape], name="flatten")

        #Fully connected layers
        self.fc6_1 = tf.contrib.layers.fully_connected(inputs=self.flatten, num_outputs=4096)
        self.fc6_2 = tf.contrib.layers.fully_connected(inputs=self.fc6_1, num_outputs=4096)
        self.fc6_3 = tf.contrib.layers.fully_connected(inputs=self.fc6_2, num_outputs=4096)
        self.logits = tf.contrib.layers.fully_connected(inputs=self.fc6_3, num_outputs=10,activation_fn=None)

        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='actual_label')

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
