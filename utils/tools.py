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
import gensim
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


def cosine(v1, v2):
    numerator = np.dot(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-7
    return numerator / denominator


def load_embedding(emb_fn):
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format(emb_fn, binary=True)

    zeros = np.zeros(w2v_model.vectors.shape[1], dtype=np.float32)
    embedding = np.insert(w2v_model.vectors, 0, zeros, axis=0)
    print("Embedding: ", embedding.shape)
    padding_value = 0
    word2index = {v:k+1 for k, v in w2v_model.index2word.items()}

    return embedding, word2index, padding_value
