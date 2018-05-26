from base.base_model import BaseModel
import tensorflow as tf


class DenseTwoLayerModel(BaseModel):
    def __init__(self, config):
        super(DenseTwoLayerModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # Network architecture
        d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, 512, activation=tf.nn.relu, name="dense2")
        logits = tf.layers.dense(d2, 10, name="out")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=logits))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
