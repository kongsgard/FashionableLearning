from base.base_model import BaseModel
import tensorflow as tf
import numpy as np


class CapsnetModel(BaseModel):
    def __init__(self, config):
        super(CapsnetModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        #Create the model
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image")
        self.x_reshaped = tf.reshape(self.x, [self.config.batch_size, 28, 28, 1], name='Reshape_input') #NOTE:Change batch_size to -1

        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="label")
        self.y_max = tf.argmax(self.y, axis=1)

        # Convolutional layer
        self.conv1 = tf.layers.conv2d(inputs = self.x_reshaped,filters = 256,kernel_size = [9,9], padding='valid',activation=tf.nn.relu)

        # Primary capsules layer
        primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV',
                                batch_size = self.config.batch_size, stddev = self.config.stddev,
                                iter_routing = self.config.iter_routing, epsilon = self.config.epsilon)
        self.caps1 = primaryCaps(self.conv1, kernel_size=9, stride=2)

        # Digit capsules layers
        digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC',
                                batch_size = self.config.batch_size, stddev = self.config.stddev,
                                iter_routing = self.config.iter_routing, epsilon = self.config.epsilon)
        self.caps2 = digitCaps(self.caps1)


        # Decoder
        self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + self.config.epsilon)
        self.softmax_v = tf.nn.softmax(self.v_length, axis=1)

        self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
        self.argmax_idx  = tf.reshape(self.argmax_idx, shape=(self.config.batch_size, ))
        self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.y, (-1, 10, 1)))
        self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + self.config.epsilon)

        vector_j = tf.reshape(self.masked_v, shape=(self.config.batch_size, -1))
        fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
        fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
        self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

        with tf.name_scope("loss"):
            self.loss()
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.cross_entropy,
                                                                                     global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.to_int32(self.y_max), self.argmax_idx)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def loss(self):
        # 1. The margin loss
        max_l = tf.square(tf.maximum(0., self.config.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - self.config.m_minus))
        max_l = tf.reshape(max_l, shape=(self.config.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(self.config.batch_size, -1))
        T_c = self.y
        L_c = T_c * max_l + self.config.lambda_val * (1 - T_c) * max_r
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.x, shape=(self.config.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        self.cross_entropy = self.margin_loss + self.config.regularization_scale * self.reconstruction_err


class CapsLayer(object):
    def __init__(self, num_outputs, vec_len, batch_size, stddev, iter_routing, epsilon, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type
        self.batch_size = batch_size
        self.stddev = stddev
        self.iter_routing = iter_routing
        self.epsilon = epsilon

    def __call__(self, input, kernel_size=None, stride=None):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                                                    self.kernel_size, self.stride, padding="VALID",
                                                    activation_fn=tf.nn.relu)

                capsules = tf.reshape(capsules, (self.batch_size, -1, self.vec_len, 1))
                capsules = squash(self, capsules)
                return(capsules)

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                self.input = tf.reshape(input, shape=(self.batch_size, -1, 1, input.shape[-2].value, 1))
                with tf.variable_scope('routing'):
                    b_IJ = tf.constant(np.zeros([self.batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self, self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)
            return(capsules)

def routing(self, input, b_IJ):
    W = tf.get_variable('Weight', shape=(1, 1152, 160, 8, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=self.stddev))
    biases = tf.get_variable('bias', shape=(1, 1, 10, 16, 1))
    input = tf.tile(input, [1, 1, 160, 1, 1])
    u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, 1152, 10, 16, 1])
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    for r_iter in range(self.iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            c_IJ = tf.nn.softmax(b_IJ, axis=2)
            if r_iter == self.iter_routing - 1:
                s_J = tf.multiply(c_IJ, u_hat)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(self,s_J)
            elif r_iter < self.iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(self, s_J)
                v_J_tiled = tf.tile(v_J, [1, 1152, 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                b_IJ += u_produce_v
    return(v_J)

def squash(self, vector):
    vec_squared_norm = tf.reduce_sum(tf.square(vector), axis = -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + self.epsilon)
    vec_squashed = scalar_factor * vector
    return(vec_squashed)
