# -*- coding: utf-8 -*-
"""
Segmenter. Transform Sentence to Words

AUTHOR: Yue Peng
EMAIL: ypeng7@outlook.com
DATE: 2018.10.02
"""
import os, sys, codecs, re, pickle
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import jieba
import jieba.posseg as pseg
import nltk
# nltk.download("stopwords")
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords as stw
from nltk.stem.porter import PorterStemmer
import langid
import logging
jieba.setLogLevel(logging.INFO)

from config import Config


config = Config()
jieba.load_userdict(config.user_dict)
stemmer = PorterStemmer()
# If you want more synonyms...
UPDATE_DICT = {}


class Segmenter(object):
    __slots__ = ("_stopwords", "_synonyms")

    def __init__(self):
        self._stopwords = [line.strip() for line in codecs.open(config.stop_words, "r", "utf-8").readlines()]
        with codecs.open(config.synonyms_path, "rb") as f:
            self._synonyms = pickle.load(f)
        self._synonyms.update(UPDATE_DICT)
        self._stopwords += stw.words("english")
        # english_punctuations
        self._stopwords += [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']

    def __str__(self):
        return "Stopwords size: {}\nSynonyms size: {}".format(len(self._stopwords), len(self._synonyms))

    __repr__ = __str__

    @property
    def stopwords(self):
        return self._stopwords

    @stopwords.setter
    def stopwords(self, value):
        if not isinstance(value, list):
            raise ValueError("Stopwords must be a list!")
        self._stopwords = value

    @property
    def synonyms(self):
        return self._synonyms
    
    @synonyms.setter
    def synonyms(self, value):
        if not isinstance(value, dict):
            raise ValueError("Synonyms must be a dictionary!")
        self._synonyms = value

    def _remove_stopwords(self, tokens):
        for t in tokens:
            if t not in self._stopwords:
                yield t

    def _remove_synonyms(self, tokens):
        for t in tokens:
            yield self._synonyms.get(t, t)

    def tokenize(self, sentence):
        text = sentence.strip().lower()
        text = re.sub(r"\D1\D", "一", text)
        text = re.sub(r"\D2\D", "二", text)
        text = re.sub(r"\D3\D", "三", text)
        text = re.sub(r"\D4\D", "四", text)
        text = re.sub(r"\D5\D", "五", text)
        text = re.sub(r"\D6\D", "六", text)
        text = re.sub(r"\D7\D", "七", text)
        text = re.sub(r"\D8\D", "八", text)
        text = re.sub(r"\D9\D", "九", text)
        text = re.sub(r"\D10\D", "十", text)
        text = re.sub(r"\D11\D", "十一", text)
        text = re.sub(r"\D12\D", "十二", text)
        text = re.sub(r"\d", "0", text)
        yield from jieba.cut(text)

    @staticmethod
    def segment_chinese_sentence(sentence):
        """Return segmented sentence.
        
        [description]
        
        Arguments:
            sentence {str} -- [description]
        """
        seg_list = jieba.cut(sentence, cut_all=False)
        seg_sentence = u" ".join(seg_list)
        return seg_sentence.strip()

    def process_sentence(self, sentence):
        assert isinstance(sentence, str), "You must enter a string!"
        """Segmenter for input sentence.
        
        [description]
        
        Arguments:
            sentence {str} -- [description]
        """
        # Chinese
        if langid.classify(sentence)[0] == "zh":
            tokens = self.tokenize(sentence)
            # Two Functional Filters
            for token_filter_func in [self._remove_stopwords, self._remove_synonyms]:
                tokens = token_filter_func(tokens)
            ret = [t for t in tokens if t != " "]
            if not ret:
                return ["UNK"]
            return ret
        # English 
        elif langid.classify(sentence)[0] == "en":
            tokens = word_tokenize(sentence)
            return [stemmer.stem(w.lower()) for w in tokens if w.lower() not in self._stopwords and w != " "]
        else:
            tokens = self.tokenize(sentence)
            for token_filter_func in [self._remove_synonyms, self._remove_stopwords]:
                tokens = token_filter_func(tokens)
            ret = [t for t in tokens if t != " "]
            if not ret:
                return ["UNK"]
            return ret 

    @staticmethod
    def all_chinese(sentence):
        """Return only chinese word
        
        [description]
        
        Arguments:
            sentence {[str]} -- [description]
        
        Returns:
            [type] -- [description]
        """
        res = ""
        for c in sentence:
            if '\u4e00' <= c <= '\u9fa5':
                res += c
            else:
                continue
        return res

    @staticmethod
    def extract_noun(sentence):
        res = []
        for i, j in pseg.cut(sentence):
            if j == "n":
                res.append(i)
        return res


def main():
    cut = Segmenter()
    print(cut.process_sentence('往饭卡里充的钱没有充进去怎么办？'))
    print(cut.process_sentence('I have a pen.'))
    print(cut.process_sentence('I have 1张饭卡.'))
    print(cut.extract_noun('往饭卡里充的钱没有充进去怎么办？'))


if __name__ == "__main__":
    main()
