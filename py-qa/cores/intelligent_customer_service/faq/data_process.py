# -*- coding: utf-8 -*-
"""
Data Processing for Online QA.

@AUTHOR: Yue Peng
@EMAIL: yuepeng@sf-express.com
@DATE: 2018.10.10
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir))
import pickle
import codecs
import pandas as pd
import gensim
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config
from cores.features.corpus import label_words_cluster
from cores.features.embedding import Embedding
from cores.utils.segmenter import Segmenter
from cores.utils.log import getLogger


config = Config()
cut = Segmenter()
logger = getLogger(__name__)


def load_data():
    with codecs.open(config.online_train, encoding="utf-8") as f:
        df_train = pd.read_csv(f, index_col=0).reset_index(drop=True)

    with codecs.open(config.online_test, encoding="utf-8") as f:
        df_test = pd.read_csv(f, index_col=0).reset_index(drop=True)

    with codecs.open(os.path.join(config.ini["dataDir"], config.ini["folders"]["online_qa"], "mapping.csv"), encoding="utf-8") as f:
        df_map = pd.read_csv(f).reset_index(drop=True)

    qs = df_train["query"].tolist()
    label = df_train["label"].tolist()
    qs_test = df_test["query"].tolist()
    label_test = df_test["label"].tolist()
    id2label = {k: v for k, v in zip(df_map["filtered_label"].tolist(), df_map["lei_label"].tolist())}
    label2id = {v: k for k, v in id2label.items()}
    del df_train, df_test, df_map

    label = [id2label[l] for l in label]
    label_test = [id2label[l] for l in label_test]
    return qs, label, qs_test, label_test, label2id, id2label


def data_pipeline(embedding=None):
    
    qs, label, qs_test, label_test, label2id, id2label = load_data()
    documents = label_words_cluster(qs, label, label2id)
    qs_seg = list(map(cut.process_sentence, qs))
    qs_test_seg = list(map(cut.process_sentence, qs_test))

    emb_model = Embedding(documents, qs_seg)
    emb_model_test = Embedding(documents, qs_test_seg)
    if embedding == "sum":
        qs_embedded = emb_model.weighted_sum_embedding()
        qs_test_embedded = emb_model_test.weighted_sum_embedding()
        y = list(map(lambda x: label2id[x], label))
        y_test = list(map(lambda x: label2id[x], label_test))
        return qs_embedded, y, qs_test_embedded, y_test

    qs_embedded = emb_model.sif_embedding()
    qs_test_embedded = emb_model_test.sif_embedding()
    y = list(map(lambda x: label2id[x], label))
    y_test = list(map(lambda x: label2id[x], label_test))

    return qs_embedded, y, qs_test_embedded, y_test


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
    _, _, _, _, label2id, _ = load_data()
    return label2id


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
        if file_type == "train":
            x, y, _, _, _, _ = load_data()
        elif file_type == "test":
            _, _, x, y, _, _ = load_data()

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
    test_dl = DataLoader(feng_test, batch_size=3036, shuffle=True, collate_fn=CollateFn())
    assert feng_train.label2id == feng_test.label2id
    return train_dl, test_dl, feng_train.label2id, embedding



if __name__ == '__main__':
    # x, y, x_test, y_test = data_pipeline("sum")
    # import lightgbm as lgb
    # train_data = lgb.Dataset(x, y)
    # gbm = lgb.train(train_set=train_data, params={"objective": "multiclass", "num_class": 132, "learning_rate": 0.01, "random_state": 257})
    _, _, label2id, _ = data_loader()
    print(len(label2id))
