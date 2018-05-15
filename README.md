# How to train a model
1. Change directory to FashionableLearning/mains/
2. `$ python [main].py -c [main-config]`

**Example**:
To train the softmax model, type `$ python softmax.py -c softmax`

# How to visualize a trained model in tensorboard
1. `$ tensorboard --logdir=[path_to_log-directory]`
2. Go to `http://localhost:6006/`

---

Project architecture based on https://github.com/MrGemy95/Tensorflow-Project-Template
