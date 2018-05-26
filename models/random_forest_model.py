from base.base_model import BaseModel
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

class RandomForestModel(BaseModel):
    def __init__(self, config):
        super(RandomForestModel, self).__init__(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        num_classes = 10
        num_features = 784
        num_trees = 10
        max_nodes = 1000

        # Input and Target data
        self.x = tf.placeholder(tf.float32, shape=[None, num_features], name="image")
        # For random forest, labels must be integers (the class id)
        self.y = tf.placeholder(tf.int32, shape=[None], name="labels")

        # Random Forest Parameters
        hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                              num_features=num_features,
                                              num_trees=num_trees,
                                              max_nodes=max_nodes).fill()

        # Build the Random Forest
        forest_graph = tensor_forest.RandomForestGraphs(hparams)

        output, _, _ = forest_graph.inference_graph(self.x)

        self.increment_global_step_op = tf.assign(self.global_step_tensor, self.global_step_tensor+1)    

        with tf.name_scope("loss"):
            # Get training graph and loss
            self.train_step = forest_graph.training_graph(self.x, self.y)
            self.loss = forest_graph.training_loss(self.x, self.y)

            correct_prediction = tf.equal(tf.argmax(output, 1), tf.cast(self.y, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
