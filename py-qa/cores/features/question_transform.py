# -*- coding: utf-8 -*-
"""
Feature Engineering Model.

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.03
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from cores.utils import log
from cores.utils.helper import log_time_delta
from config import Config

logger = log.getLogger(__name__)
config = Config()


@log_time_delta
def load_embedding():
    """
    loading pre-trained embedding
    :return: class<gensim.models.keyedvectors.Word2VecKeyedVectors>
    """
    return gensim.models.KeyedVectors.load_word2vec_format(
        config.embedding_path, binary=True)


def is_chinese(c):
    """check if char is chinese or not
    
    [description]
    
    Arguments:
        c -- single chinese character 
    """
    if '\u4e00' <= c <= '\u9fa5':
        return True
    else:
        return False


def bigram(sentence):
    """ Bi-gram Feature word
    
    [description]
    
    Arguments:
        sentence -- [description]
    """
    start, end = 0, 0
    segs = []
    precedentByChinese = False
    while (end < len(sentence)):
        if sentence[end].isspace():
            segs.append(sentence[start:end])
            while (end < len(sentence) and sentence[end].isspace()):
                end += 1
            if end >= len(sentence):
                break
            start = end

        if is_chinese(sentence[end]):
            if end > start:
                segs.append(sentence[start:end])
                precedentByChinese = False
            else:
                precedentByChinese = True
            start = end

            if (start < len(sentence) - 1 and is_chinese(sentence[start+1])):
                segs.append(sentence[start:start+2])
                start += 1
                end = start
            else:
                if not precedentByChinese:
                    segs.append(sentence[start:start+1])
                start += 1
                end = start
        else:
            end += 1

    return segs
