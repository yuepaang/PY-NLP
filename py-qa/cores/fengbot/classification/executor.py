# -*- coding: utf-8 -*-
"""
AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.08.03
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)))
from cores.fengbot.classification.features import FeatureVector
from cores.fengbot.classification.model import MLP1, MLP2, data_preprocessing
from cores.utils.tools import softmax
from config import Config
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

config = Config()


class TextClassification(object):
    """丰声机器人粗分类（4类/42类）
    4类
    训练正确率: 98.97% (98.75%)
    测试正确率: 99.13% (98.33%)
    42类
    训练正确率: 99.62% (98.51%)
    测试正确率: 94.71% (93.46%)
    """
    def __init__(self, task=None):
        assert task == "4" or task == "42", "We only got task 4 and 42~~~"
        self.task = task
        self.feature_vector = FeatureVector(task=self.task)
        if self.task == "4":
            params = {"input_size": 300, "hidden1_size": 1024, "hidden2_size": 128, "hidden3_size": 32, "num_classes": 4}
            self.model = MLP1(**params)
            self.model.load_state_dict(torch.load(os.path.join(config.ini["modelDir"], config.ini["models"]["c4"]), map_location='cpu'))
            self.model.eval()
            self.pair_dict = dict(zip(range(len(self.feature_vector.encoder.classes_)), self.feature_vector.encoder.classes_.tolist()))

        elif self.task == "42":
            params = {"input_size": 300, "hidden1_size": 1024, "hidden2_size": 256, "num_class": 42}
            self.model = MLP2(**params)
            self.model.load_state_dict(torch.load(os.path.join(config.ini["modelDir"], config.ini["models"]["c42"]), map_location='cpu'))
            self.model.eval()
            self.pair_dict = dict(zip(range(len(self.feature_vector.encoder.classes_)), self.feature_vector.encoder.classes_.tolist()))

    def eval(self):
        """
            Print out our model training and testing accufacy.
        """
        train_emb, train_labels, test_emb, test_labels, _, _ = self.feature_vector.load_formatted()
        output_train = np.asarray(self.model(torch.from_numpy(train_emb).float()).data)
        output_test = np.asarray(self.model(torch.from_numpy(test_emb).float()).data)

        probs_train = softmax(output_train)
        predicted_label_train = np.argmax(probs_train, axis=1)
        probs_test = softmax(output_test)
        predicted_label_test = np.argmax(probs_test, axis=1)

        print("Our training accuracy is %.4f"% (sum(np.equal(train_labels, predicted_label_train)) / train_emb.shape[0]))

        print("Our test accuracy is %.4f" % (sum(np.equal(test_labels, predicted_label_test)) / test_emb.shape[0]))

    def predict(self, question):
        vec = self.feature_vector.question_transform(question)
        output = np.asarray(self.model(torch.from_numpy(vec).float()).data)
        return self.pair_dict[np.argmax(softmax(output), axis=1)[0]]


class Executor(object):
    def __init__(self, task):
        assert task == "4" or task == "42", "We only got task 4 and 42~~~"
        self.task = task
        self.qs_embedded, self.y, self.qs_test_embedded, self.y_test, self.id2label = data_preprocessing(task=self.task)

        if self.task == "4":
            params = {"input_size": 300, "hidden1_size": 1024, "hidden2_size": 256, "hidden3_size": 64, "num_classes": 4}
            self.model = MLP1(**params)
            self.model.load_state_dict(torch.load(os.path.join(config.ini["modelDir"], config.ini["models"]["c4_new"]), map_location='cpu'))
            self.model.eval()
        elif self.task == "42":
            params = {"input_size": 300, "hidden1_size": 512, "hidden2_size": 128, "num_class": 42}
            self.model = MLP2(**params)
            self.model.load_state_dict(torch.load(os.path.join(config.ini["modelDir"], config.ini["models"]["c42_new"]), map_location='cpu'))
            self.model.eval()

    def eval(self):
        logits_train = self.model(torch.from_numpy(self.qs_embedded).float())
        pred_train = logits_train.argmax(dim=-1)
        num_right_train = (pred_train == torch.LongTensor(self.y)).sum().item()
        num_total_train = self.qs_embedded.shape[0]
        acc_train = num_right_train / float(num_total_train)

        logits_test = self.model(torch.from_numpy(self.qs_test_embedded).float())
        pred_test = logits_test.argmax(dim=-1)
        num_right_test = (pred_test == torch.LongTensor(self.y_test)).sum().item()
        num_total_test = self.qs_test_embedded.shape[0]
        acc_test = num_right_test / float(num_total_test)

        print("Our training accuracy is %.4f" % acc_train)
        print("Our test accuracy is %.4f" % acc_test)


def main():
    executor = Executor(task="42")
    executor.eval()


if __name__ == "__main__":
    main()
