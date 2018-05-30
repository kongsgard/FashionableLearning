import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for self.cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        Implement the logic of epoch:
        - Loop over the number of iterations in the config and call the train step
        - Add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        Implement the logic of the train step:
        - Run the tensorflow session
        - Return any metrics you need to summarize
        """
        raise NotImplementedError
