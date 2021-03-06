from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class CNNThreeConvLayersTrainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(CNNThreeConvLayersTrainer, self).__init__(sess, model, data, config,logger)

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
        print("Epoch#:", self.cur_epoch)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        train_summaries_dict = {
            'loss': train_loss,
            'acc': train_acc,
            'histogram_convlayer1_weight': self.sess.run(self.model.w1),
            'histogram_convlayer1_bias': self.sess.run(self.model.b1),
            'histogram_convlayer2_weight': self.sess.run(self.model.w2),
            'histogram_convlayer2_bias': self.sess.run(self.model.b2),
            'histogram_convlayer3_weight': self.sess.run(self.model.w3),
            'histogram_convlayer3_bias': self.sess.run(self.model.b3),
            'histogram_fc_weight': self.sess.run(self.model.w4),
            'histogram_fc_bias': self.sess.run(self.model.b4),
            'histogram_readout_weight': self.sess.run(self.model.w5),
            'histogram_readout_bias': self.sess.run(self.model.b5),
        }

        valid_loss, valid_acc = self.valid_step()
        valid_summaries_dict = {
            'loss': valid_loss,
            'acc': valid_acc,
        }

        if self.cur_epoch == self.config.num_epochs:
            test_loss, test_acc = self.test_step()
            test_summaries_dict = {
                'test_loss': test_loss,
                'test_acc': test_acc,
            }
            self.logger.summarize(cur_it, summaries_dict=test_summaries_dict, summarizer='test')

        self.logger.summarize(cur_it, summaries_dict=train_summaries_dict, summarizer='train')
        self.logger.summarize(cur_it, summaries_dict=valid_summaries_dict, summarizer='valid')
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.keep_prob: 0.3,
                     self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy,
                                      self.model.accuracy], feed_dict=feed_dict)
        return loss, acc

    def valid_step(self):
        x_valid, y_valid = self.data.get_valid_data()
        feed_dict = {self.model.x: x_valid, self.model.y: y_valid,self.model.keep_prob: 1.,
                    self.model.is_training: False}
        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                   feed_dict=feed_dict)
        return loss, acc

    def test_step(self):
        x_test, y_test = self.data.get_test_data()
        feed_dict = {self.model.x: x_test, self.model.y: y_test, self.model.keep_prob: 1.,
                     self.model.is_training: False}
        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                   feed_dict=feed_dict)
        return loss, acc
