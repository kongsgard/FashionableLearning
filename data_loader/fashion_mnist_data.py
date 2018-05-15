import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

fashion_mnist = input_data.read_data_sets('data/fashion', source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/', one_hot=True)

class DataGenerator:
    def __init__(self, config):
        self.config = config

        # Import data
        self.input = fashion_mnist.test.images
        self.y = fashion_mnist.test.labels

    def next_batch(self, batch_size):
        yield fashion_mnist.train.next_batch(self.config.batch_size)
        #idx = np.random.choice(500, batch_size)
        #yield self.input[idx], self.y[idx]
