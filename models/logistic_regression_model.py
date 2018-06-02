from base.base_model import BaseModel
import tensorflow as tf
import os


class LogisticRegressionModel(BaseModel):
    def __init__(self, config):
        super(LogisticRegressionModel, self).__init__(config)
        self.build_model()
        self.init_saver()


    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # Placeholders
        self.x = tf.placeholder(tf.float32, [None, 784], name="image")
        self.y = tf.placeholder(tf.float32, [None, 10], name="label")

        # Weights and bias
        self.w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
        self.b = tf.Variable(tf.zeros([10]), name='bias')

        # Construct model
        self.logits = tf.matmul(self.x, self.w) + self.b

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.logits)
                )
            self.train_step = tf.train.AdamOptimizer(0.5).minimize(self.cross_entropy, global_step=self.global_step_tensor)

            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.logits, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
