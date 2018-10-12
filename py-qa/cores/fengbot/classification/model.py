# -*- coding: utf-8 -*-
"""
Model for classification.

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.03
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
from cores.fengbot.dataset.data_helper import data_extract
from cores.features.corpus import label_words_cluster
from cores.features.embedding import Embedding
from cores.utils.segmenter import Segmenter
from cores.utils.tools import log_time_delta
from config import Config
from cores.utils import log

config = Config()
cut = Segmenter()
logger = log.getLogger(__name__)


@log_time_delta
def data_preprocessing(task="42"):
    qs, label, qs_test, label_test, label2id, id2label = data_extract(task=task)
    qs_seg = list(map(cut.process_sentence, qs))
    qs_test_seg = list(map(cut.process_sentence, qs_test))
    documents = label_words_cluster(sentences=qs, labels=label, label2id=label2id)
    emb_model = Embedding(documents, qs_seg)
    emb_model_test = Embedding(documents, qs_test_seg)
    # qs_embedded = emb_model.sif_embedding()
    # qs_test_embedded = emb_model_test.sif_embedding()
    qs_embedded = emb_model.weighted_sum_embedding()
    qs_test_embedded = emb_model_test.weighted_sum_embedding()
    y = list(map(lambda x: label2id[x], label))
    y_test = list(map(lambda x: label2id[x], label_test))
    return qs_embedded, y, qs_test_embedded, y_test, id2label


def data_loader(x, y, batch_size):
    xt = torch.from_numpy(x).float()
    yt = torch.LongTensor(y)
    data_set = TensorDataset(xt, yt)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


class MLP1(nn.Module):
    """classification for 4 classes
    
    [description]
    """
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, num_classes):
        super(MLP1, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, num_classes)
        self.weight_init()

    def weight_init(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.constant_(self.fc1.bias, 0)
        init.normal_(self.fc2.bias, 0)
        init.normal_(self.fc3.bias, 0)
        init.normal_(self.fc4.bias, 0)

    def forward(self, x):
        out = self.dropout(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


class MLP2(nn.Module):
    """classification for 42 classes
    
    [description]
    """
    def __init__(self, input_size, hidden1_size, hidden2_size, num_class):
        super(MLP2, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_class)
        self.weight_init()
    def weight_init(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.normal_(self.fc1.bias)
        init.normal_(self.fc2.bias)
        init.normal_(self.fc3.bias)
    
    def forward(self, x):
        out = self.dropout(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def train(task):
    train_emb, train_labels, test_emb, test_labels, _ = data_preprocessing(task=task)
    input_size = 300
    hidden1_size = 1024
    hidden2_size = 256
    hidden3_size = 64
    num_classes = 4
    batch_size = 128
    learning_rate = 1e-4
    num_epochs = 5000
    model = MLP1(input_size, hidden1_size, hidden2_size, hidden3_size, num_classes)
    # params = {"input_size": 300, "hidden1_size": 512, "hidden2_size": 128, "num_class": 42}
    # model = MLP2(**params)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_loss = 0
    cnt = 0
    acc_flag = 0.984
    for epoch in range(num_epochs):
        if epoch > 0:
            model.eval()
            for x, y in data_loader(test_emb, test_labels, test_emb.shape[0]):
                logits = model(x)
            pred = logits.argmax(dim=-1)
            num_right_test = (pred == y).sum().item()
            num_total_test = x.size(0)
            acc = num_right_test / float(num_total_test)
            logger.warn("The test accuracy of epoch %d is %.4f." % (epoch + 1, acc))
            if acc > acc_flag:
                acc_flag = acc
                torch.save(model.state_dict(), "{}/epoch{}_{:.4f}.pt".format(config.ini["modelDir"], epoch + 1, acc))
                logger.warn("saved_models: {}/epoch{}_{:.4f}.pt".format(config.ini["modelDir"], epoch + 1, acc))
            model.train()

        total_right = 0
        total = 0
        for i, (x, y) in enumerate(data_loader(train_emb, train_labels, batch_size)):
            optimizer.zero_grad()
            outputs = model(x)
            pred = outputs.argmax(dim=-1)
            num_right_train = (pred == y).sum().item()
            total_right += num_right_train
            total += x.size(0)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            cnt += 1
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Loss: {:.4f}".format(epoch, (total_loss / (cnt * batch_size))))
        logger.warn("Training Acc: {:.4f}".format(total_right / float(total)))


if __name__ == '__main__':
    train(task="4")
