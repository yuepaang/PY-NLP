# -*- coding: utf-8 -*-
"""
Data Pipeline for PyTorch.

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.03
"""
import os, sys, pickle, codecs
import gensim
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from cores.utils import log
from cores.dataset.data_helper import load_dataset
from config import Config
from cores.utils.segmenter import Segmenter

logger = log.getLogger(__name__)
config = Config()
cut = Segmenter()


def load_embedding():
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(config.embedding, binary=True)
    word2idx = {}
    zeros = np.zeros(word2vec.vectors.shape[1], dtype=np.float32)
    embedding = np.insert(word2vec.vectors, 0, zeros, axis=0)
    print("embedding shape: ", embedding.shape)
    padding_value = 0

    for i, w in enumerate(word2vec.index2word):
        word2idx[w] = i + 1
    return embedding, word2idx, padding_value


def to_index(question, word2idx, unknown="UNK"):
    l = []
    words = cut.process_sentence(question) if cut.process_sentence(question) else ["UNK"]
    # if not words:
    #     l.append(word2idx[unknown])
    #     return l
    for w in words:
        try:
            l.append(word2idx[w])
        except KeyError as e:
            logger.info(e)
            l.append(word2idx[unknown])
    return l


def label2id():
    x, y = load_dataset(file_type="train")
    return dict((k, v) for k, v in zip(set(y), range(len(set(y)))))


class FengBotDataset(Dataset):
    def __init__(self, file_type, word2idx, padding_value, transform=None):
        super(FengBotDataset, self).__init__()
        self.word2idx = word2idx
        self.padding_value = padding_value
        self._label2id = label2id()
        self.load_data(file_type)
        self.transform = transform

        if self.transform is None:
            self.transform = lambda x: x

    @property
    def label2id(self):
        return self._label2id

    def load_data(self, file_type):
        data = []
        x, y = load_dataset(file_type=file_type)
        y = list(map(lambda x: self._label2id[x], y))
        for i, j in zip(x, y):
            qidxs = to_index(i, self.word2idx)
            data.append((qidxs, j))
        self.data = np.array(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        data = self.data[index]
        return self.transform(data[0]), data[1]


class CollateFn(object):
    def __init__(self, max_len=None, padding_value=0):
        self.max_len = max_len
        self.padding_value = padding_value
    
    def __call__(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        seqs, labels = zip(*batch)
        lengths = list(map(len, seqs))
        max_len = self.max_len if self.max_len is not None else len(seqs[0])

        padded = []
        for seq in seqs:
            # 重要的话说三次
            tmp = seq + seq + seq
            if len(tmp) < max_len:
                padded.append(tmp + [self.padding_value] * (max_len - len(tmp)))
            else:
                padded.append(tmp[:max_len])
        return padded, labels, lengths


def data_loader():
    embedding, word2idx, padding_value = load_embedding()
    feng_train = FengBotDataset("train", word2idx, padding_value) 
    feng_test = FengBotDataset("test", word2idx, padding_value)
    train_dl = DataLoader(feng_train, batch_size=int(config.ini["hyparams"]["batch_size"]), shuffle=True, collate_fn=CollateFn())
    test_dl = DataLoader(feng_test, batch_size=2630, shuffle=True, collate_fn=CollateFn())
    assert feng_train.label2id == feng_test.label2id
    return train_dl, test_dl, feng_train.label2id, embedding


if __name__ == "__main__":
    _, _, label2id, _ = data_loader()
    print(len(label2id))
