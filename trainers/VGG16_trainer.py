from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class VGG16Trainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(VGG16Trainer, self).__init__(sess, model, data, config,logger)

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

        print("Train acc:", acc) # TODO: Remove
        print("Train loss:", loss) # TODO: Remove

        cur_it = self.model.global_step_tensor.eval(self.sess)
        train_summaries_dict = {
            'loss': train_loss,
            'acc': train_acc,
        }


        loop = tqdm(range(self.config.num_iter_per_epoch_test))
        losses = []
        accs = []
        for _ in loop:
            loss, acc = self.test_step()
            losses.append(loss)
            accs.append(acc)
        test_loss = np.mean(losses)
        test_acc = np.mean(accs)

        print("Test acc:", acc) # TODO: Remove
        print("Test loss:", loss) # TODO: Remove

        test_summaries_dict = {
            'loss': test_loss,
            'acc': test_acc,
        }

        self.logger.summarize(cur_it, summaries_dict=train_summaries_dict, summarizer='train')
        self.logger.summarize(cur_it, summaries_dict=test_summaries_dict, summarizer='test')
        #self.model.save(self.sess)


    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def test_step(self):
        x_test, y_test = next(self.data.next_batch_testdata(self.config.batch_size))
        feed_dict = {self.model.x: x_test, self.model.y: y_test,self.model.is_training: False}
        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                   feed_dict=feed_dict)
        return loss, acc
