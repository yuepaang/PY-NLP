# -*- coding: utf-8 -*-
"""
Corpus engineering.

AUTHOR: Yue Peng
EMAIL: ypeng7@outlook.com
DATE: 2018.10.09
"""
import os
import sys
from copy import deepcopy
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from config import Config
from utils import log
from utils.segmenter import Segmenter
from utils.tools import log_time_delta


config = Config()
cut = Segmenter()
logger = log.getLogger(__name__)


@log_time_delta
def label_words_cluster(sentences, labels, label2id):
    """words clustering based on labels
    
    [description]
    
    Arguments:
        sentences {[type]} -- [description]
        labels {[type]} -- [description]
    """
    x = deepcopy(sentences)
    y = deepcopy(labels)
    # Transform sentence into words
    for i, s in enumerate(x):
        x[i] = cut.process_sentence(s) if cut.process_sentence(s) else ["UNK"]
    # word_clusters = [set() for _ in range(len(set(y)))]
    word_clusters = [[] for _ in range(len(label2id))]
    for j, l in enumerate(y):
        for w in x[j]:
            # word_clusters[label2id[l]].add(w)
            word_clusters[label2id[l]].append(w)
    topic_docs = list(map(lambda x: " ".join(x), word_clusters))
    return topic_docs


def overlap(query_seg, answers_base):
    """Compute the overlap feature
    
    # TODO Sparse if answers base too large
    
    Arguments:
        query_seg {list<str>} -- segmented query
        answers_base {list<str>} -- list of answers
    """
    if not query_seg:
        return [0]
    features = []
    for a in answers_base:
        cnt = 0
        for w in query_seg:
            if w in a:
                cnt += 1
        features.append(cnt / len(query_seg))
    return features
