# -*- coding: utf-8 -*-
"""
Simple Logging Service

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.02
"""
import os, sys, time
from functools import wraps
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from config import Config
import logging

config = Config()

formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s - %(message)s", datefmt='%H:%M:%S')
log_file = "%s/%s.log" % (config.log_path, time.strftime("%Y%b%d"))

print("********Saving logs into", log_file, "************")

fh = logging.FileHandler(log_file)
fh.setFormatter(formatter)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
ch.setLevel(logging.WARN)


def getLogger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger


if __name__ == "__main__":
    logger = getLogger('root')
    logger.info('start to log...')
