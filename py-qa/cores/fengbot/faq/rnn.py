# -*- coding: utf-8 -*-
"""
633 faq
AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.03

Epoch: 271, Test Acc: 0.8167
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
from cores.dataset.pipeline import data_loader
from config import Config
from cores.utils import log

logger = log.getLogger(__name__)
config = Config()


class RNN(nn.Module):
    def __init__(self, embedding, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding), freeze=True)
        self.fc1 = nn.Linear(embedding.shape[1], hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.fc2 = torch.nn.utils.weight_norm(nn.Linear(hidden_size, num_classes))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, x_lens):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True)
        _, (h, c) = self.lstm(x, None)
        hidden = h[0]
        x = self.dropout(hidden)
        logits = self.fc2(x)
        return logits


def train():
    print("Start training ...")
    total_loss = 0
    num_total = 0
    num_right = 0
    step = 0
    acc_flag = 0
    for e in range(int(config.ini["hyparams"]["num_epochs"])):
        if e % 10 == 1:
            model.eval()
            for x, y, lens in test_dl:
                x = torch.tensor(x).to(device)
                y = torch.tensor(y).to(device)
                lens = torch.tensor(lens).to(device)
                logits = model(x, lens)
                pred = logits.argmax(dim=-1)
                num_right_test = (pred == y).sum().item()
                num_total_test = x.size(0)
                acc = num_right_test / float(num_total_test)
                if acc > acc_flag and acc > 0.81:
                    acc_flag = acc
                    logger.warn("The best accuracy hits %.4f" % acc)
                    torch.save(model.state_dict(), config.saved_models + "/model-%s.pt" % round(acc, 4))
            print("=============Epoch: {}, Test Acc: {:.4} =================".format(e, acc))
        model.train()
        for x, y, lens in train_dl:
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)
            lens = torch.tensor(lens).to(device)

            logits = model(x, lens)
            pred = logits.argmax(dim=-1)
            loss = criterion(logits, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            avg_loss = total_loss / float(step)

            num_right += (pred == y).sum().item()
            num_total += x.size(0)
            acc = num_right / float(num_total)

            if step % 200 == 0:
                print("Epoch: {}, Step: {}, Acc: {:.4f}, Loss: {:.4f}".format(e, step, acc, avg_loss))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.warn("*************We are using GPU to train...*************")
    train_dl, test_dl, label2id, embedding = data_loader()
    model = RNN(embedding=embedding, hidden_size=int(config.ini["nnparams"]["hidden_size"]), num_layers=int(config.ini["nnparams"]["num_layers"]), num_classes=int(config.ini["nnparams"]["num_classes"]))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.ini["hyparams"]["learning_rate"]))
    train()
