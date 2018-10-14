#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset, Estimator API of Tensorflow

# @Date    : 2018-10-13
# @Author  : Yue Peng (ypeng7@outlook.com)
"""
import os
import sys
import tensorflow as tf
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


def train_input_fn():
    x = tf.data.Dataset.from_tensor_slices()