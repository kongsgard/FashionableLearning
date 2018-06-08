from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import resources

class RandomForestTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(RandomForestTrainer, self).__init__(sess, model, data, config, logger)
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), resources.initialize_resources(resources.shared_resources()))
        self.sess.run(self.init)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
        train_loss = np.mean(losses)
        train_acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        train_summaries_dict = {
            'loss': train_loss,
            'acc': train_acc,
        }

        test_loss, test_acc = self.test_step()
        test_summaries_dict = {
            'loss': test_loss,
            'acc': test_acc,
        }

        self.logger.summarize(cur_it, summaries_dict=train_summaries_dict, summarizer='train')
        self.logger.summarize(cur_it, summaries_dict=test_summaries_dict, summarizer='test')
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        batch_y = np.argmax(batch_y, axis=1) # Decode one-hot labels
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc, _ = self.sess.run([self.model.train_step, self.model.loss,
                                      self.model.accuracy, self.model.increment_global_step_op], feed_dict=feed_dict)
        return loss, acc

    def test_step(self):
        x_test, y_test = self.data.get_test_data()
        y_test = np.argmax(y_test, axis=1) # Decode one-hot labels
        feed_dict = {self.model.x: x_test, self.model.y: y_test, self.model.is_training: False}
        loss, acc, _ = self.sess.run([self.model.loss, self.model.accuracy, self.model.increment_global_step_op],
                                   feed_dict=feed_dict)
        return loss, acc
