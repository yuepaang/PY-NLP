# -*- coding: utf-8 -*-
"""
Distributed Representation for Sentence.

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.08
"""
import os, sys
import numpy as np
import gensim
from copy import deepcopy
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from config import Config
from cores.utils import log
from cores.utils.segmenter import Segmenter
from cores.utils.tfidf import word_idf, sif_embedding, get_weighted_average


config = Config()
cut = Segmenter()
logger = log.getLogger(__name__)


class Embedding(object):
    def __init__(self, documents, data_seg):
        # default pre-trained word embedding
        self._word_embedding = gensim.models.KeyedVectors.load_word2vec_format(config.embedding, binary=True)
        self.documents = documents
        self.data_seg = data_seg
        self.weight_of_word = word_idf(self.documents)
        # initized some components for sentence embedding
        self.word2idx, self.wv_mat = None, None
        self._word2vec()
        self.sentence_words_weights = None
        self._sentece_words_weights()
        self.data_seg_index = None
        self._data_to_index()

    @property
    def word_embedding(self):
        return self._word_embedding

    @word_embedding.setter
    def word_embedding(self, value):
        self._word_embedding = value

    def _word2vec(self):
        """initialize the word embedding matrix
        
        [description]
        """
        self.word2idx = dict((v, k) for k, v in enumerate(self._word_embedding.index2word))
        self.wv_mat = np.zeros([len(self.word2idx), self._word_embedding.vector_size])
        for k, v in self.word2idx.items():
            self.wv_mat[v, :] = self._word_embedding.get_vector(k)

    def _sentece_words_weights(self):
        self.sentence_words_weights = []
        for i, q in enumerate(self.data_seg):
            if not q:
                logger.warning("Index %s" % i)
                self.sentence_words_weights.append(np.array([0]))
                continue
            sentence_weight = np.zeros([len(q)])
            for i, w in enumerate(q):
                sentence_weight[i] = self.weight_of_word.get(w, 0.0)
            self.sentence_words_weights.append(sentence_weight)

    def _data_to_index(self):
        self.data_seg_index = []
        for word_list in self.data_seg:
            res = []
            if not word_list:
                self.data_seg_index.append(self.word2idx["UNK"])
                continue
            for w in word_list:
                res.append(self.word2idx.get(w, self.word2idx["UNK"]))
            self.data_seg_index.append(res)

    def sif_embedding(self):
        embeddings = get_weighted_average(self.wv_mat, self.data_seg_index, self.sentence_words_weights)
        return sif_embedding(embeddings)

    @staticmethod
    def padding(data_seg, max_len, type="repeat"):
        data = deepcopy(data_seg)
        if type == "repeat":
            padded_dat = [0] * len(data)
            for i, d in enumerate(data):
                if not d:
                    padded_dat[i] = ["UNK"] * max_len
                    continue
                while len(d) > 0 and len(d) < max_len:
                    d += d
                    if len(d) >= max_len:
                        padded_dat[i] = d[:max_len]
                padded_dat[i] = d[:max_len]
        elif type == "unk":
            padded_dat = [0] * len(data)
            for i, d in enumerate(data):
                if not d:
                    padded_dat[i] = ["UNK"] * max_len
                    continue
                if len(d) > 0 and len(d) < max_len:
                    d += ["UNK"] * (max_len - len(d))
                padded_dat[i] = d[:max_len]
        return padded_dat

    def weighted_sum_embedding(self, pad="repeat"):
        if pad == "repeat" or "unk":
            data = self.padding(self.data_seg, int(config.ini["corpus"]["max_length"]), type=pad)
        else:
            data = deepcopy(self.data_seg)
        vectors = []
        for _, s in enumerate(data):
            word_vector = np.zeros([1, 300])
            for w in s:
                try:
                    word_vector += self.word_embedding.get_vector(w) * self.weight_of_word.get(w, 1.0)
                except KeyError as e:
                    logger.info(e)
                    word_vector += self.word_embedding.get_vector("UNK")
            vectors.append(word_vector)
        return np.array(vectors).reshape([len(data), -1])


def main(argv=None):
    from cores.dataset.data_helper import data_extract
    from cores.features.corpus import label_words_cluster
    qs, label, qs_test, label_test, label2id, id2label = data_extract(task="4")
    documents = label_words_cluster(qs+qs_test, label+label_test, label2id)
    from cores.utils.segmenter import Segmenter
    cut = Segmenter()
    qs_seg = list(map(lambda x: cut.process_sentence(x), qs))
    emb = Embedding(documents, qs_seg)
    print(emb.sif_embedding().shape)
    print(emb.weighted_sum_embedding().shape)


if __name__ == '__main__':
    main()