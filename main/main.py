import tensorflow as tf
import multiprocessing

import _init_paths
from data_loader.fashion_mnist_data import DataGenerator
from utils.config import process_config, import_class
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args

from models.logistic_regression_model import LogisticRegressionModel
from trainers.logistic_regression_trainer import LogisticRegressionTrainer

def main():
    # Capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

        Model = import_class(args.config, "model")
        Trainer = import_class(args.config, "trainer")
    except:
        print("Missing or invalid arguments")
        exit(0)

    # Create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # Create a tensorflow session
    sess = tf.InteractiveSession()
    # Create an instance of the model specified
    model = Model(config)
    # Load model if it exists
    model.load(sess)
    # Create the data generator
    data = DataGenerator(config)
    # Create tensorboard logger
    logger = Logger(sess, config)
    # Create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, data, config, logger)

    # Train the model
    trainer.train()
    print("Training complete!")
    sess.close()

if __name__ == '__main__':
    p = multiprocessing.Process(target=main)
    p.start()
    p.join()
