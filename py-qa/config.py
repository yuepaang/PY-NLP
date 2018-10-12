# -*- coding: utf-8 -*-
"""
config module to load configurations

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.02
"""
import sys, os, codecs
import configparser
from time import localtime, strftime
from cores.utils.helper import singleton

CONF_DIR = os.path.dirname(os.path.realpath(__file__))


def get_cfg_dir():
    """Get config dir
    
    [description]
    """
    if not os.path.exists(CONF_DIR):
        os.mkdir(CONF_DIR)
    return CONF_DIR


def get_cfg_path(filename):
    """Get config path
    
    [description]
    
    Arguments:
        filename {str} -- [description]
    """
    return os.path.join(get_cfg_dir(), filename)


def load_config(filename):
    """Load configurations
    
    [description]
    
    Arguments:
        filename {str} -- [description]
    """
    cf = get_cfg_path(filename)
    if not os.path.exists(cf):
        f = codecs.open(cf, "w")
        f.close()

    config = configparser.ConfigParser()
    config.read(cf)
    return config


def read_properties(filename="config.ini"):
    """Read Properties from Config File.
    
    [description]
    
    Keyword Arguments:
        filename {str} -- [description] (default: {"config.ini"})
    """
    config = load_config(filename)
    secs = config.sections()
    conf = {}
    # config2dict
    for x in secs:
        conf[x] = {y: config.get(x, y) for y in config.options(x)}
    conf["rootDir"] = CONF_DIR
    conf["dataDir"] = os.path.join(CONF_DIR, "data")
    conf["modelDir"] = os.path.join(CONF_DIR, "saved_models")
    return conf


@singleton
class Config(object):
    """docstring for Config"""
    def __init__(self, ):
        self.ini = read_properties()
        self.root_dir = self.ini["rootDir"]
        self.config_ini_path = get_cfg_path(filename="config.ini")
        self.data_dir = self.ini["dataDir"]
        self.user_dict = os.path.join(self.ini["dataDir"], self.ini["data"]["user_dict_fn"])
        self.embedding = os.path.join(self.ini["dataDir"], self.ini["data"]["embedding_fn"])
        self.stop_words = os.path.join(self.ini["dataDir"], self.ini["data"]["stopwords_fn"])
        self.train_path = os.path.join(self.ini["dataDir"], self.ini["data"]["train_data_fn"])
        self.test_path = os.path.join(self.ini["dataDir"], self.ini["data"]["test_data_fn"])
        self.hash_4 = os.path.join(self.ini["dataDir"], self.ini["data"]["hash_4_fn"])
        self.hash_42 = os.path.join(self.ini["dataDir"], self.ini["data"]["hash_42_fn"])
        self.hash_4_folder = os.path.join(self.ini["dataDir"], self.ini["folders"]["class_4"])
        self.hash_42_folder = os.path.join(self.ini["dataDir"], self.ini["folders"]["class_42"])
        self.hash_633_folder = os.path.join(self.ini["dataDir"], self.ini["folders"]["class_633"])
        self.vocab_path = os.path.join(self.ini["dataDir"], self.ini["data"]["vocab_fn"])
        self.synonyms_path = os.path.join(self.ini["dataDir"], self.ini["data"]["synonyms_fn"])
        self.log_path = os.path.join(CONF_DIR, self.ini["log"]["log_folder"])

        self.online_train = os.path.join(self.ini["dataDir"], self.ini["folders"]["online_qa"], self.ini["data"]["online_train"])
        self.online_test = os.path.join(self.ini["dataDir"], self.ini["folders"]["online_qa"], self.ini["data"]["online_test"])


if __name__ == "__main__":
    config = Config()
    print("rootDir %s" % config.root_dir)
    print("dataDir %s" % config.data_dir)
    print(config.config_ini_path)
