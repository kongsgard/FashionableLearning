import tensorflow as tf

import _init_paths
from data_loader.fashion_mnist_data import DataGenerator
from models.simple_CNN_model import SimpleCNNModel
from trainers.simple_CNN_trainer import SimpleCNNTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args


def main():
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("Missing or invalid arguments")
        exit(0)

    # Create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # Create tensorflow session
    sess = tf.Session()
    # Create an instance of the model you want
    model = SimpleCNNModel(config)
    # Load model if exists
    model.load(sess)
    # Create your data generator
    data = DataGenerator(config)
    # Create tensorboard logger
    logger = Logger(sess, config)
    # Create trainer and pass all the previous components to it
    trainer = SimpleCNNTrainer(sess, model, data, config, logger)

    # Train the model
    trainer.train()


if __name__ == '__main__':
    main()
