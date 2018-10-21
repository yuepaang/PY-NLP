# -*- coding: utf-8 -*-
# @Author: Yue Peng
# @Email: yuepaang@gmail.com
# Date: Oct 21, 2018
# Created on: 22:11:16
import os
import sys
import codecs
from gensim.models.word2vec import Word2Vec
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from config import Config

config = Config()


class MySentence(object):
    """Already space-separated Sentence txt file Handler

    """
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        for line in codecs.open(self.file_path, "r", "utf-8"):
            words = line.split(" ")
            result_word = []
            for word in words:
                if word and word != "\n":
                    result_word.append(word)
            yield result_word


def main():
    trainable = False
    txt_path = config.ini["dataDir"] + "/case.txt"
    file_path = config.ini["dataDir"] + "/case_embedding.bin"
    sentences = MySentence(file_path=txt_path)
    model = Word2Vec(sentences, workers=4, size=200)
    if trainable is True:
        model.save(file_path)
    else:
        model.save_word2vec_format(file_path, binary=True)


if __name__ == '__main__':
    main()
