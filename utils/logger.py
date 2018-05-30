import tensorflow as tf
import os


class Logger:
    def __init__(self, sess,config):
        self.sess = sess
        self.config = config
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"),
                                                          self.sess.graph)
        self.valid_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "valid"))
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "test"))

    # it can summarize scalars and images.
    def summarize(self, step, summarizer="train", scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param summarizer: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        #summary_writer = self.train_summary_writer if summarizer == "train" else self.valid_summary_writer
        if summarizer == "train":
            summary_writer = self.train_summary_writer
        elif summarizer == "valid":
            summary_writer = self.valid_summary_writer
        else:
            summary_writer = self.test_summary_writer


        with tf.variable_scope(scope):
            if summaries_dict is not None and summarizer is not "test":
                summary_list = []
                for tag, value in summaries_dict.items():
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = tf.placeholder('float32', value.shape, name=tag)
                            self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_placeholders[tag] = tf.placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                            self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag])

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()

            # Logging test accuracy and loss
            if summarizer is "test":
                summary_list2 = []
                self.summary_placeholders2 = {}
                self.summary_ops2 = {}
                for tag, value in summaries_dict.items():
                    self.summary_placeholders2[tag] = tf.placeholder('string', value.shape, name=tag)
                    self.summary_ops2[tag] = tf.summary.text(tag, self.summary_placeholders2[tag])
                    summary_list2.append(self.sess.run(self.summary_ops2[tag], {self.summary_placeholders2[tag]: str(value)}))
                for summary in summary_list2:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()
