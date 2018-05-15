from base.base_model import BaseModel
import tensorflow as tf


class SoftmaxModel(BaseModel):
    def __init__(self, config):
        super(SoftmaxModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        self.y = tf.matmul(self.x, W) + b

        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 10])

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy, global_step=self.global_step_tensor)

            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
