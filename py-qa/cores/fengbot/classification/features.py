# -*- coding: utf-8 -*-
"""
Feature Engineering of Current Model

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.07.31
"""
import os, sys
import time
import codecs
import numpy as np
import gensim
from sklearn.preprocessing import LabelEncoder
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from collections import Counter
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))
from cores.utils.segmenter import Segmenter
from cores.fengbot.dataset.data_helper import data_extract
from config import Config

config = Config()


class FeatureVector(object):
    """Feature Engineering Part for Text Classification
    
    [description]
    
    Attributes:
        print("Done!"): [description]
    """
    def __init__(self, task=None):
        self.pool = ThreadPool(os.cpu_count())
        self.encoder = LabelEncoder()
        self.cut = Segmenter() 
        self.emb_model = gensim.models.KeyedVectors.load_word2vec_format(config.embedding, binary=True)
        # void init
        self.vectorizer = None
        self.l2id = None
        self.questions, self.labels, self.questions_test, self.labels_test = None, None, None, None
        self.topic_corpus, self.word_scores, self.train_words, self.test_words = None, None, None, None
        self.train_x_aug, self.test_x_aug, self.train_emb, self.test_emb = None, None, None, None
        self.train_labels, self.test_labels = None, None
        # initialization based on task
        if task is not None:
            self.init_data(task=task)
            if self.train_words:
                self.corpus = [w for l in self.train_words for w in l]
            self.word_counter = Counter(self.corpus)
            # most common 4000 words
            self.word_counter = dict(self.word_counter.most_common(4000))
            self.vocab = list(self.word_counter.keys())
            self.vocab += ["UNK"]
            print("The number of words in our vocabulary is %d\n" % (len(self.vocab)))
            self.word2id = {w: i for i, w in enumerate(self.vocab)}
            self.id2word = {i: w for i, w in enumerate(self.vocab)}

            self.init_vector(task=task)
            print("Successfully initialized word embedding!\n")

    @staticmethod
    def _analyze_length(num, word_list):
        """
            找出超过最大长度的句子
            :return: list[idx]
        """
        len_stat = [len(v) for v in word_list]
        return np.where(np.array(len_stat) > num)[0].tolist()


    def __del__(self):
        self.pool.close()
        self.pool.join()

    def __word_cut(self, word_list):
        return list(map(self.cut.process_sentence, word_list))

    def load_formatted(self):
        return self.train_emb, self.train_labels, self.test_emb, self.test_labels, self.word_scores, self.encoder

    def init_data(self, task="4"):
        # if task == "4":
        self.questions, self.labels, self.questions_test, self.labels_test, _, _ = data_extract(task=task)
        # self.topic_corpus, _ = self.topic_aggregate(self.questions, self.labels, duplicated=True)
        self.topic_corpus, self.l2id = self.topic_aggregate(self.questions, self.labels, duplicated=True)
        self.word_scores, self.vectorizer = self.tfidf_extract(self.topic_corpus)
        self.train_words, self.test_words = self.pool.map(self.__word_cut, [self.questions, self.questions_test])
        self.train_x_aug, self.test_x_aug = self.pool.map(self.dup3, [self.train_words, self.test_words])

        # elif task == "42":
        #     self.questions, self.labels, self.questions_test, self.labels_test = load_data_second()
        #     # 0927
        #     # self.topic_corpus, _ = self.topic_aggregate(self.questions, self.labels, duplicated=True)
        #     self.topic_corpus, self.l2id = self.topic_aggregate(self.questions, self.labels, duplicated=False)
        #     self.word_scores, self.vectorizer = self.tfidf_extract(self.topic_corpus)
        #     self.train_words, self.test_words = self.pool.map(self.__word_cut, [self.questions, self.questions_test])
        #     self.train_x_aug, self.test_x_aug = self.pool.map(self.dup3, [self.train_words, self.test_words])

    def init_vector(self, task="4"):
        if task == "4":
            self.train_emb, self.test_emb = self.pool.map(partial(self.embedding_transform, word_scores=self.word_scores), [self.train_x_aug, self.test_x_aug])
            self.train_labels = self.encoder.fit_transform(self.labels)
            self.test_labels = self.encoder.transform(self.labels_test)

        elif task == "42":
            self.train_emb, self.test_emb = self.pool.map(partial(self.embedding_transform, word_scores=self.word_scores), [self.train_x_aug, self.test_x_aug])
            self.train_labels = self.encoder.fit_transform(self.labels)
            self.test_labels = self.encoder.transform(self.labels_test)

    def embedding_transform(self, data, word_scores):
        emb = self.word_embedding_sum(self.emb_model, data, tfidf=word_scores)
        return np.array(emb).reshape([len(emb), -1])

    def question_transform(self, question):
        res = self.cut.process_sentence(question)
        res = self.dup3([res])
        res = self.word_embedding_sum(self.emb_model, res, self.word_scores)
        return np.array(res).reshape((1, -1))

    def topic_aggregate(self, x, y, duplicated=False, level="word"):
        """根据类别形成topic

        用空格分开并以主题集合, 去掉非中文，（英文少有keyword）

        Args:
            x: list[str]
            y: list[str]

        Returns:
            list with size = # of classes
            list
        """
        _x = deepcopy(x)
        _y = deepcopy(y)
        for i, q in enumerate(_x):
            _x[i] = self.remove_stop_words(self.cut.process_sentence(self.cut.all_chinese(q)))
        num_class = len(set(y))
        class2id = dict(zip(list(set(y)), range(num_class)))
        if level == "char":
            corpus_topic = [[] for _ in range(num_class)]
            for idx, c in enumerate(_y):
                for w in _x[idx]:
                    corpus_topic[class2id[c]].add(w)
            topic_docs = list(map(lambda j: "".join(j), corpus_topic))
            return topic_docs, class2id

        if not duplicated:
            corpus_topic = [set() for _ in range(num_class)]
            for idx, c in enumerate(_y):
                for w in _x[idx]:
                    corpus_topic[class2id[c]].add(w)
            topic_docs = list(map(lambda j: " ".join(list(j)), corpus_topic))
        else:
            corpus_topic = [[] for _ in range(num_class)]
            for idx, c in enumerate(_y):
                corpus_topic[class2id[c]].append(" ".join(_x[idx]))
            topic_docs = list(map(lambda j: " ".join(j), corpus_topic))
        return topic_docs, class2id

    @staticmethod
    def tfidf_extract(corpus, level=None):
        """calculate tfidf score for words

        [description]

        Args:
            corpus: list[list[str]]
            level: [description] (default: {None})

        Returns:
            tfidf score for each word
            dict[float]
        """
        if level == "word":
            vectorizer = TfidfVectorizer(analyzer='word', max_features=5000)
        elif level == "ngram":
            vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=5000)
        elif level == "char":
            vectorizer = TfidfVectorizer(encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer='char', stop_words=None, ngram_range=(2, 4), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=True)
        else:
            vectorizer = TfidfVectorizer()
        vectorizer.fit(corpus)
        idf = vectorizer.idf_
        word_tfidf = dict(zip(vectorizer.get_feature_names(), idf))
        return word_tfidf, vectorizer

    @staticmethod
    def remove_stop_words(words):
        """remove stop words

        [description]

        Args:
            words: [description]
        """
        stop_words = [line.strip() for line in codecs.open(config.stop_words, "r", "utf-8").readlines()]
        return [w for w in words if w not in stop_words]

    def word_embedding_sum(self, emb_model, word_list, tfidf=None):
        """句子向量的表示

        词向量的（TFIDF权重）和

        Args:
            emb_model: gensim model
            word_list: list[list[str]]
            tfidf: dict[float] (default: {None})

        Returns:
            sentence representation
            list[numpy array with shape (1, vector_size)]
        """
        embed = []
        for _, v in enumerate(word_list):
            word_vec = np.zeros([1, 300])
            for w in v:
                if w not in self.vocab:
                    word_vec += emb_model.get_vector("UNK")
                    continue
                try:
                    if not tfidf:
                        word_vec += emb_model.get_vector(w)
                    elif tfidf:
                        word_vec += emb_model.get_vector(w) * tfidf.get(w, 1.0)
                except KeyError:
                    word_vec += emb_model.get_vector("UNK")
            embed.append(word_vec)
        return embed

    @staticmethod
    def __dup3(vec, max_length):
        vec_ = deepcopy(vec)
        while len(vec_) < max_length:
            vec_ += vec_
            if len(vec_) >= max_length:
                return vec_[:max_length]
        return vec_[:max_length]

    def dup3(self, word_list):
        """Repeated Padding
        
        repeats all the words until the length of word list hit the maximum length
        
        Args:
            word_list: a list of words list
            max_length: integer
        
        Returns:
            a list of padding word list
            list[list]
        """
        return list(map(partial(self.__dup3, max_length=int(config.ini["corpus"]["max_length"])), word_list))

    print("Done!")
