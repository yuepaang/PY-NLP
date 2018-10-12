# -*- coding: utf-8 -*-
"""
Classical MLP for Online QA.

@AUTHOR: Yue Peng
@EMAIL: yuepeng@sf-express.com
@DATE: 2018.10.10
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from config import Config
from cores.utils.log import getLogger
from cores.faq.data_process import data_pipeline

config = Config()
logger = getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import TensorDataset, DataLoader


def data_loader(x, y, batch_size):
    xt = torch.from_numpy(x).float()
    yt = torch.LongTensor(y)
    data_set = TensorDataset(xt, yt)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        self.weight_init()

    def weight_init(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.constant_(self.fc1.bias, 0)
        init.normal_(self.fc2.bias, 0)
        init.normal_(self.fc3.bias, 0)

    def forward(self, x):
        out = self.dropout(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.warn("*************We are using GPU to train...*************")
    else:
        logger.warn("*************We are using CPU to train...*************")
    input_size = 300
    hidden1_size = 1024
    hidden2_size = 512
    num_classes = 132
    batch_size = 256
    # FIXIT lr might be too high
    learning_rate = 0.001
    num_epochs = 5000
    model = MLP(input_size, hidden1_size, hidden2_size, num_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    total_loss = 0
    cnt = 0
    acc_flag = 0.8632
    for epoch in range(num_epochs):
        if epoch > 0:
            model.eval()
            for x, y in data_loader(test_emb, test_labels, len(test_labels)):
                x = x.to(device)
                y = y.to(device)
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
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            pred = outputs.argmax(dim=-1)
            num_right_train = (pred == y).sum().item()
            total_right += num_right_train
            total += x.size(0)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            cnt += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Loss: {:.4f}".format(epoch, (total_loss / (cnt * batch_size))))
        logger.warn("Training Acc: {:.4f}".format(total_right / float(total)))

if __name__ == '__main__':
    train_emb, train_labels, test_emb, test_labels = data_pipeline("sum")
    train()
