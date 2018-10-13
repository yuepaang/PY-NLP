# -*- coding: utf-8 -*-
"""
TF Keras Model for classification.

AUTHOR: Yue Peng
EMAIL: ypeng7@outlook.com
DATE: 2018.10.13
"""
import sys, os
import tensorflow as tf
import time
from sklearn.preprocessing import LabelBinarizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))
from config import Config
from cores.fengbot.classification.model import data_preprocessing
from cores.utils.log import getLogger

config = Config()
logger = getLogger(__name__)

lb = LabelBinarizer()
qs_embedded, y, qs_test_embedded, y_test, id2label = data_preprocessing(task="42")

lb.fit(range(max(y) + 1))
y_oh = lb.transform(y)
y_test_oh = lb.transform(y_test)


def train_input_fn():
    global qs_embedded, y_oh
    x = tf.cast(qs_embedded, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y_oh))
    dataset = dataset.repeat(500)
    dataset = dataset.batch(512)
    return dataset


def test_input_fn():
    global qs_test_embedded, y_test_oh
    x = tf.cast(qs_test_embedded, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y_test_oh))
    dataset = dataset.repeat(1)
    dataset = dataset.batch(qs_test_embedded.shape[0])
    return dataset


def mlp_model():
    input_layer = tf.keras.layers.Input(shape=(300, ))
    use_bias = False
    dropout = tf.keras.layers.Dropout(0.5)(input_layer)
    # Layer 1
    h1 = tf.keras.layers.Dense(1024, activation="relu")(dropout)
    # Layer 2
    h2 = tf.keras.layers.Dense(512, activation="relu")(h1)
    # Layer 3
    logits = tf.keras.layers.Dense(42, activation="softmax", name="output")(h2)

    return tf.keras.Model(inputs=input_layer, outputs=logits)


model = mlp_model()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])


strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=1)
config_tf = tf.estimator.RunConfig(train_distribute=strategy)

est = tf.keras.estimator.model_to_estimator(keras_model=model, config=config_tf, model_dir="py")
est = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir="py")


class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)


time_hist = TimeHistory()
est.train(train_input_fn, hooks=[time_hist])
est.evaluate(test_input_fn)
