# -*- coding: utf-8 -*-
"""
We want the intrinsic similarity 

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.09
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import numpy as np


def proximity_matrix(model, X, normalize=True):
    """Proximity Help for Random Forest Model
    
    [description]
    
    Arguments:
        model -- [description]
        X -- [description]
    
    Keyword arguments:
        normalize -- [description] (default: {True})
    """

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:, 0]
    proxMat = 1 * np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:, i]
        proxMat += 1 * np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat

