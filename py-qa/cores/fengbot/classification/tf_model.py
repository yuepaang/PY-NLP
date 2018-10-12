# -*- coding: utf-8 -*-
"""
TF version Model for classification.

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.03
"""
import sys, os
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))
from config import Config
from cores.fengbot.classification.model import data_preprocessing

config = Config()

qs_embedded, y, qs_test_embedded, y_test, id2label = data_preprocessing(task="42")

# =================
#  tf Dataset API
# =================
train_data = tf.data.Dataset.from_tensor_slices(qs_embedded)
train_label = tf.data.Dataset.from_tensor_slices(y).map(lambda x: tf.one_hot(x, 42))
test_data = tf.data.Dataset.from_tensor_slices(qs_test_embedded)
test_label = tf.data.Dataset.from_tensor_slices(y_test).map(lambda x: tf.one_hot(x, 42))

trainDataSet = tf.data.Dataset.zip((train_data, train_label)).shuffle(5000).batch(128)
testDataSet = tf.data.Dataset.zip((train_data, train_label)).batch(qs_test_embedded.shape[0])

iterator = tf.data.Iterator.from_structure(trainDataSet.output_types, trainDataSet.output_shapes)
train_init_op = iterator.make_initializer(trainDataSet)
test_init_op = iterator.make_initializer(testDataSet)

next_ele = iterator.get_next()
# ======================
# Model Part
# ======================
class MLP(object):
    def __init__(self, next_element=None):
        self.lr = 1e-4
        self._init_weights()
        self.x = next_element[0]
        self.y = next_element[1]
        self.logits = self.output(self.x)
        self.loss = self.loss_func()
        self.train_op = self.train()
        self.preds = tf.argmax(self.logits, axis=1)
        self.corrects = tf.equal(tf.cast(self.preds, tf.int64), tf.argmax(self.y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.corrects, tf.int64), name="accuracy")

    def _init_weights(self):
        with tf.name_scope("hidden1"):
            self.w1 = tf.get_variable(name="w1", shape=[300, 1024], initializer=tf.glorot_normal_initializer(seed=7), dtype = tf.float64)
            self.b1 = tf.get_variable(name="b1", shape=[1024], initializer=tf.truncated_normal_initializer(stddev=0.001), dtype = tf.float64)
        with tf.name_scope("hidden2"):
            self.w2 = tf.get_variable(name="w2", shape=[1024, 256], initializer=tf.glorot_normal_initializer(seed=7), dtype = tf.float64)
            self.b2 = tf.get_variable(name="b2", shape=[256], initializer=tf.truncated_normal_initializer(stddev=0.001), dtype=tf.float64)
        with tf.name_scope("hidden3"):
            self.w3 = tf.get_variable(name="w3", shape=[256, 42], initializer=tf.glorot_normal_initializer(seed=7),dtype=tf.float64)
            self.b3 = tf.get_variable(name="b3", shape=[42], initializer=tf.truncated_normal_initializer(stddev=0.001), dtype=tf.float64)

    def output(self, x):
        a1 = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
        a2 = tf.nn.relu(tf.matmul(a1, self.w2) + self.b2)
        with tf.name_scope("output"):
            logits = tf.matmul(a2, self.w3) + self.b3
        return logits

    def loss_func(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y))

    def train(self):
        return tf.train.AdamOptimizer(self.lr).minimize(self.loss)


if __name__ == "__main__":
    with tf.Session() as sess:
        model = MLP(next_element=next_ele)
        sess.run(tf.global_variables_initializer())

