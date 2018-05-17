import json
from bunch import Bunch
from importlib import import_module
import os
import re


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # Parse the configurations from the config json file provided
    json_file = "../configs/" + json_file + ".json"
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # Convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join("../experiments", config.exp_name, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint/")
    return config

def import_class(config, object):
    module = import_module(object + "s." + config + "_" + object)
    name = re.sub(r'[^a-zA-Z0-9]', '', config) # Remove non-alphanumeric characters, including '_'
    class_name = name[:1].upper() + name[1:] + object.capitalize()
    return getattr(module, class_name)
