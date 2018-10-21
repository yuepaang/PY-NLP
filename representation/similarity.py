# -*- coding: utf-8 -*-
# @Author: Yue Peng
# @Email: yuepaang@gmail.com
# Date: Oct 21, 2018
# Created on: 00:58:40
import os
import sys
import gensim
import jieba
from distance import levenshtein
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from config import Config

config = Config()

jieba.load_userdict(config.ini["dataDir"]+"/"+config.ini["data"]["user_dict_fn"])


class TextDistance(object):
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2
        self.model_fn = config.ini["dataDir"]+"/"+config.ini["data"]["embedding_fn"]
        self.model = gensim.models.KeyedVectors.load_word2vec_format(self.model_fn)

    def edit_distance(self):
        """Edit Distance
        
        Returns:
            int -- [description]
        """

        return levenshtein(self.s1, self.s2)

    @staticmethod
    def _add_space(s):
        return " ".join(list(s))

    def jaccard_similarity(self):
        """Jaccard Index
        
        Returns:
            float -- Larger value, more similar
        """

        s1, s2 = self._add_space(self.s1), self._add_space(self.s2)
        cv = CountVectorizer(tokenizer=lambda s: s.split())
        corpus = [s1, s2]
        vectors = cv.fit_transform(corpus).toarray()
        # intersaction
        numerator = np.sum(np.min(vectors, axis=0))
        # union
        denominator = np.sum(np.max(vectors, axis=0))
        return 1.0 * numerator / denominator
    
    def tf_similarity(self):
        """Cosine Similarity of two TF vector
        
        Returns:
            float -- [description]
        """

        s1, s2 = self._add_space(self.s1), self._add_space(self.s2)
        cv = CountVectorizer(tokenizer=lambda s: s.split())
        corpus = [s1, s2]
        vectors = cv.fit_transform(corpus).toarray()
        numerator = np.dot(vectors[0], vectors[1])
        denominator = np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
        return numerator / denominator
    
    def tfidf_similarity(self):
        """Cosine Similarity of two TF-IDF vector
        
        Returns:
            float -- [description]
        """

        s1, s2 = self._add_space(self.s1), self._add_space(self.s2)
        tv = TfidfVectorizer(tokenizer=lambda s: s.split())
        corpus = [s1, s2]
        vectors = tv.fit_transform(corpus).toarray()
        numerator = np.dot(vectors[0], vectors[1])
        denominator = np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
        return numerator / denominator
    
    def vector_similarity(self):
        """Word2Vec Cosine Similarity
        
        Returns:
            float -- [description]
        """
        v1, v2 = self._sentence_vector(self.s1), self._sentence_vector(s2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def _sentence_vector(self, s):
        words = jieba.lcut(s)
        v = np.zeros(64)
        for word in words:
            v += self.model[word]
        v /= len(words)
        return v


def proximity_matrix(model, X, normalize=True):
    """Proximity Help for Random Forest Model
    
    [description]
    
    Arguments:
        model -- [description]
        X -- [description]
    
    Keyword arguments:
        normalize -- [description] (default: {True})
    """

    terminals = model.apply(X)
    nTrees = terminals.shape[1]

    a = terminals[:, 0]
    proxMat = 1 * np.equal.outer(a, a)

    for i in range(1, nTrees):
        a = terminals[:, i]
        proxMat += 1 * np.equal.outer(a, a)

    if normalize:
        proxMat = proxMat / nTrees

    return proxMat


def main():
    s1 = "你在干嘛呢"
    s2 = "你在干什么呢"
    td = TextDistance(s1, s2)
    print(td.edit_distance())
    print(td.jaccard_similarity())
    print(td.tf_similarity())
    print(td.tfidf_similarity())
    print(td.vector_similarity())


if __name__ == '__main__':
    main()
