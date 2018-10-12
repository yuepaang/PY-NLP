# -*- coding: utf-8 -*-
"""
Some Tools

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.02
"""
import sys, os
from functools import wraps
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from utils import log

logger = log.getLogger(__name__)


def log_time_delta(func):
    """Print the running time for the function
    
    [description]
    
    Decorators:
        wraps
    
    Arguments:
        func {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        # print("%s has run %.2f seconds." % (func.__name__, delta))
        logger.warn("%s has run %.2f seconds." % (func.__name__, delta))
        return ret
    return _deco


def softmax(x):
    """Calculate the softmax of a 2-dim np array
    
    [description]
    
    Arguments:
        x {np.array} -- [description]
    
    Returns:
        [np.array] -- [description]
    """
    assert len(x.shape) == 2, "Input array's dimension must be equal to 2"
    s = np.max(x, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div
