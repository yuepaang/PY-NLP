#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset, Estimator API of Tensorflow

# @Date    : 2018-10-13
# @Author  : Yue Peng (ypeng7@outlook.com)
"""
import sys, os
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))
from config import Config
from utils.log import getLogger


config = Config()
logger = getLogger(__name__)

# TODO:
# writer = tf.python_io.TFRecordWriter("%s.tfrecord" % "test")

# features = {}
# for i in range(len(qs_embedded)):
#     features["matrix"] = tf.train.Feature(float_list=tf.train.FloatList(value=qs_embedded[i].reshape(-1)))


def train_input_fn(features, labels, batch_size, epoch):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=len(features))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def eval_input_fn(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(len(features))
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def model_fn(features, labels, mode):
    x = tf.layers.dropout(inputs=features, rate=0.5, name="dropout1")
    x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu, name="dense1")
    x = tf.layers.dense(inputs=x, units=512, activation=tf.nn.relu, name="dense2")
    logits = tf.layers.dense(inputs=x, units=42, name="output")
    predictions = {
        "classes": tf.argmax(input=logits, axis=1, name="classes"),
        "probabilities": tf.nn.softmax(logits=logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=42)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits, scope="LOSS")

    accuracy, update_op = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name="accuracy")
    batch_acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.int64), predictions["classes"]), tf.float32))
    tf.summary.scalar("batch_acc", batch_acc)
    tf.summary.scalar("streaming_acc", update_op)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": (accuracy, update_op)
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir="./py/")

classifier.train(input_fn=train_input_fn)

eval_res = classifier.evaluate(input_fn=eval_input_fn)
