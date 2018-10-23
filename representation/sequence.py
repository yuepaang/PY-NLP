# -*- coding: utf-8 -*-
# @Author: Yue Peng
# @Email: yuepaang@gmail.com
# Date: Oct 21, 2018
# Created on: 00:58:52
import numpy as np


class WordSequence(object):
    """Sentence Encoder(to index)
    
    [description]
    """
    PAD_TAG = '<PAD>'
    UNK_TAG = '<UNK>'
    START_TAG = '<BOS>'
    END_TAG = '<EOS>'
    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self):
        self.dictionary = {WordSequence.PAD_TAG: WordSequence.PAD, WordSequence.UNK_TAG: WordSequence.UNK, WordSequence.START_TAG: WordSequence.START, WordSequence.END_TAG: WordSequence.END}
        self.fitted = False

    def to_index(self, word):
        assert self.fitted, "WordSequence has not fitted"
        if word in self.dictionary:
            return self.dictionary[word]
        return WordSequence.UNK

    def to_word(self, index):
        assert self.fitted, "WordSequence has not fitted"
        for k, v in self.dictionary.items():
            if v == index:
                return k
        return WordSequence.UNK_TAG

    def _size(self):
        assert self.fitted, "WordSequence has not fitted"
        return len(self.dictionary) + 1

    def __len__(self):
        return self._size()

    def fit(self, sentences, min_count=None, max_count=None, max_features=None):
        """Train WordSequence
        
        [description]
        
        Arguments:
            sentences -- [description]
        
        Keyword arguments:
            min_count -- [description] (default: {5})
            max_count -- [description] (default: {None})
            max_features -- [description] (default: {None})
        """
        assert not self.fitted, "WordSequence can only be trained once"
        count = {}
        for sentence in sentences:
            for a in sentence:
                if a not in count:
                    count[a] = 0
                count[a] += 1

        if min_count is not None:
            count = {k:v for k, v in count.items() if v >= min_count}

        if max_count is not None:
            count = {k:v for k, v in count.items() if v <= max_count}

        if isinstance(max_features, int):
            count = sorted(list(count.items()), key=lambda x:x[1])
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
            for w, _ in count:
                self.dictionary[w] = len(self.dictionary)
        else:
            for w in sorted(count.keys()):
                self.dictionary[w] = len(self.dictionary)

        self.fitted = True

    def transform(self, sentence, max_len=None):
        assert self.fitted, "WordSequence has not fitted"

        if max_len is not None:
            r = [WordSequence.PAD] * max_len
        else:
            r = [WordSequence.PAD] * len(sentence)

        for index, a in enumerate(sentence):
            if max_len is not None and index >= len(r):
                break
            r[index] = self.to_index(a)

        return np.array(r)

    def inverse_transform(self, indices, ignore_pad=False, ignore_unk=False, ignore_start=False, ignore_end=False):
        ret = []
        for i in indices:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and ignore_end:
                continue
            ret.append(word)

        return ret


class CharSequence(object):
    """Sentence Encoder(to index)
    
    [description]
    """
    PAD_TAG = '<PAD>'
    UNK_TAG = '<UNK>'
    START_TAG = '<BOS>'
    END_TAG = '<EOS>'
    PAD = 2
    UNK = 3
    START = 0
    END = 1

    def __init__(self):
        self.dictionary = {CharSequence.PAD_TAG: CharSequence.PAD, CharSequence.UNK_TAG: CharSequence.UNK, CharSequence.START_TAG: CharSequence.START, CharSequence.END_TAG: CharSequence.END}
        self.fitted = False

    def to_index(self, word):
        assert self.fitted, "CharSequence has not fitted"
        if word in self.dictionary:
            return self.dictionary[word]
        return CharSequence.UNK

    def to_word(self, index):
        assert self.fitted, "CharSequence has not fitted"
        for k, v in self.dictionary.items():
            if v == index:
                return k
        return CharSequence.UNK_TAG

    def _size(self):
        assert self.fitted, "CharSequence has not fitted"
        return len(self.dictionary) + 1

    def __len__(self):
        return self._size()

    def fit(self, sentences, min_count=None, max_count=None, max_features=None):
        """Train CharSequence
        
        [description]
        
        Arguments:
            sentences -- list<str>
        
        Keyword arguments:
            min_count -- [description] (default: {5})
            max_count -- [description] (default: {None})
            max_features -- [description] (default: {None})
        """
        assert not self.fitted, "CharSequence can only be trained once"
        count = {}
        for sentence in sentences:
            for a in sentence:
                if a not in count:
                    count[a] = 0
                count[a] += 1

        if min_count is not None:
            count = {k:v for k, v in count.items() if v >= min_count}

        if max_count is not None:
            count = {k:v for k, v in count.items() if v <= max_count}

        if isinstance(max_features, int):
            count = sorted(list(count.items()), key=lambda x:x[1])
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
            for w, _ in count:
                self.dictionary[w] = len(self.dictionary)
        else:
            for w in sorted(count.keys()):
                self.dictionary[w] = len(self.dictionary)

        self.fitted = True

    def transform(self, sentence, max_len=None):
        """Transform sentence into indices
        
        Arguments:
            sentence {str} -- [description]
        
        Keyword Arguments:
            max_len {int} -- [description] (default: {None})
        
        Returns:
            [numpy.array] -- [description]
        """

        assert self.fitted, "CharSequence has not fitted"

        if max_len is not None:
            r = [CharSequence.PAD] * max_len
        else:
            r = [CharSequence.PAD] * len(sentence)

        for index, a in enumerate(sentence):
            if max_len is not None and index >= len(r):
                break
            r[index] = self.to_index(a)

        return np.array(r)

    def inverse_transform(self, indices, ignore_pad=False, ignore_unk=False, ignore_start=False, ignore_end=False):
        ret = []
        for i in indices:
            word = self.to_word(i)
            if word == CharSequence.PAD_TAG and ignore_pad:
                continue
            if word == CharSequence.UNK_TAG and ignore_unk:
                continue
            if word == CharSequence.START_TAG and ignore_start:
                continue
            if word == CharSequence.END_TAG and ignore_end:
                continue
            ret.append(word)

        return ret


class BagOfWord(object):
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2
    
    def add_sentence(self, sentence):
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def main():
    ws = WordSequence()
    ws.fit([
        ['第', '一', '句', '话'],
        ['第', '二', '句', '话']
    ])
    print(ws.dictionary)
    indice = ws.transform(['第', '三'])
    print(indice)

    back = ws.inverse_transform(indice)
    print(back)


    ws = CharSequence()
    ws.fit([
        '第一句话',
        '第二句话'
    ])
    print(ws.dictionary)
    indice = ws.transform('第三')
    print(indice)

    back = ws.inverse_transform(indice)
    print(back)


if __name__ == '__main__':
    main()
