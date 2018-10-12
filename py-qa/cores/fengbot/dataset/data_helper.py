# -*- coding: utf-8 -*-
"""
Data Helper file for pipeline.

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.03
"""
import sys, os, codecs, pickle
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
from config import Config
from cores.utils import log
from cores.utils.tools import log_time_delta

config = Config()
logger = log.getLogger(__name__)


def load_dataset(folder, file_type="train"):
    if file_type == "train":
        with codecs.open("{}/{}".format(folder, config.ini["data"]["qs"]), "rb") as f:
            qs = pickle.load(f)
        with codecs.open("{}/{}".format(folder, config.ini["data"]["label"]), "rb") as f:
            label = pickle.load(f)
    elif file_type == "test":
        with codecs.open("{}/{}".format(folder, config.ini["data"]["qs_test"]), "rb") as f:
            qs = pickle.load(f)
        with codecs.open("{}/{}".format(folder, config.ini["data"]["label_test"]), "rb") as f:
            label = pickle.load(f)
    return qs, label


@log_time_delta
def data_extract(task="4"):
    """extract data based on classification task
    
    Keyword Arguments:
        task {str} -- [description] (default: {"4"})
    
    Returns:
        [type] -- [description]
    """
    if task == "4":
        qs, label = load_dataset(folder=config.hash_4_folder, file_type="train")
        qs_test, label_test = load_dataset(folder=config.hash_4_folder, file_type="test")

        label2id = dict((k, v) for k, v in zip(set(label), range(len(set(label)))))
        id2label = dict((v, k) for k, v in label2id.items())
        logger.info("Successfully loaded the dataset...")

    elif task == "42":
        qs, label = load_dataset(folder=config.hash_42_folder, file_type="train")
        qs_test, label_test = load_dataset(folder=config.hash_42_folder, file_type="test")

        label2id = dict((k, v) for k, v in zip(set(label), range(len(set(label)))))
        id2label = dict((v, k) for k, v in label2id.items())
        logger.info("Successfully loaded the dataset...")

    elif task == "633":
        qs, label = load_dataset(folder=config.hash_633_folder, file_type="train")
        qs_test, label_test = load_dataset(folder=config.hash_633_folder, file_type="test")
        label2id = dict((k, v) for k, v in zip(set(label), range(len(set(label)))))
        id2label = dict((v, k) for k, v in label2id.items())

    return qs, label, qs_test, label_test, label2id, id2label


def main(argv=None):
    qs, _, _, _, label2id, _ = data_extract()
    print(qs[0])
    print(len(qs))
    print(len(label2id))


if __name__ == '__main__':
    main()
