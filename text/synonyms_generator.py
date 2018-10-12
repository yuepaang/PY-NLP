# -*- coding: utf-8 -*-
"""
Synonyms Generator based on dataset we owned.

AUTHOR: Yue Peng
EMAIL: ypeng7@outlook.com
DATE: 2018.10.02
"""
import os, sys, codecs, pickle
import synonyms
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from config import Config
from utils import log

config = Config()
logger = log.getLogger(__name__)


# generate vocabulary
def generate_vocab(text_data):
    """
        Extract all the noun from dataset
    :param text: list<str>
    :return:
    """

    vocab = set()
    for sentence in text_data:
        segs = synonyms.seg(sentence)
        words = [x for x, y in zip(segs[0], segs[1]) if y == "n"]
        for word in words:
            if len(word) >= 2:
                vocab.add(word)

    if os.path.exists(config.vocab_path):
        os.remove(config.vocab_path)

    logger.info("Writing the vocab...")
    with codecs.open(config.vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    del vocab
    logger.warning("vocab.pkl: %s" % config.vocab_path)
    print("Done!")


def generate_synonyms():
    with codecs.open(config.vocab_path, "rb") as f:
        vocab = pickle.load(f)
    synonyms_dict = {}
    for word in vocab:
        for k in synonyms.nearby(word)[0][1:5]:
            synonyms_dict[k] = word
    logger.info("Writing the synonyms...")
    with codecs.open(config.synonyms_path, "wb") as f:
        pickle.dump(synonyms_dict, f)
    logger.info("synonyms.pkl: %s" % config.synonyms_path)
    del synonyms_dict, vocab
    print("Done!")


if __name__ == "__main__":
    if not os.path.exists(config.vocab_path):
        generate_vocab()
    else:
        logger.info("%s has already existed." % config.vocab_path)
    if not os.path.exists(config.synonyms_path):
        generate_synonyms()
    else:
        logger.info("%s has already existed." % config.synonyms_path)
