# -*- coding: utf-8 -*-
"""
Configuration Module.

AUTHOR: Yue Peng
EMAIL: ypeng7@outlook.com
DATE: 2018.10.02
"""
import sys, os, codecs
# sys.path.append(os.path.dirname(__file__))
import configparser
from time import localtime, strftime
from utils.helper import Singleton

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
    conf["logDir"] = os.path.join(CONF_DIR, "logs")
    return conf


@Singleton
class Config(object):
    """docstring for Config"""
    def __init__(self, ):
        self.ini = read_properties()
        self.root_dir = self.ini["rootDir"]
        self.data_dir = self.ini["dataDir"]
        self.log_dir = self.ini["logDir"]
        self.config_ini_path = get_cfg_path(filename="config.ini")

        self.user_dict = os.path.join(self.data_dir, self.ini["data"]["user_dict_fn"])
        self.stop_words = os.path.join(self.data_dir, self.ini["data"]["stopwords_fn"])
        self.synonyms_path = os.path.join(self.data_dir, self.ini["data"]["synonyms_fn"])
        self.noun_path = os.path.join(self.data_dir, self.ini["data"]["noun_fn"])
        self.log_name = os.path.join(self.log_dir, self.ini["log"]["log_fn"])


if __name__ == "__main__":
    config = Config()
    print("rootDir %s" % config.root_dir)
    print("dataDir %s" % config.data_dir)
    print(config.config_ini_path)
