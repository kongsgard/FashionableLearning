import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

fashion_mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)

class DataGenerator:
    def __init__(self, config):
        self.config = config

    def next_batch(self, batch_size):
        yield fashion_mnist.train.next_batch(self.config.batch_size)

    def next_batch_validdata(self, batch_size):
        yield fashion_mnist.validation.next_batch(self.config.batch_size)

    def next_batch_testdata(self, batch_size):
        yield fashion_mnist.test.next_batch(self.config.batch_size)

    def get_test_data(self):
        return fashion_mnist.test.images, fashion_mnist.test.labels

    def get_valid_data(self):
        return fashion_mnist.validation.images, fashion_mnist.validation.labels
