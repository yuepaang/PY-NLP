# -*- coding: utf-8 -*-
"""
Loaded Data and Pickle them.

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.02
"""
import os, sys, codecs, pickle
import pandas as pd
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from cores.utils.helper import log_time_delta
from config import Config
from cores.utils import log

logger = log.getLogger(__name__)
config = Config()


@log_time_delta
def load_relations_first():
    """标准分类与粗分类映射

    633 -> 4

    """
    with codecs.open(config.hash_4, "r", encoding="utf-8") as f:
        _ = f.readline()
        relation_dict = {}
        for i, line in enumerate(f.readlines()):
            relation_dict[line.strip().split(",")[3]] = line.strip().split(",")[1]
    return relation_dict


@log_time_delta
def load_relations_second():
    """标准分类与42分类映射

    633 -> 42
    """
    df = pd.read_csv(config.hash_42)
    standard_question = df["标准问题"].tolist()
    first_label = df["一级标签"].tolist()
    relation_dict = {}
    for i in range(df.shape[0]):
        relation_dict[standard_question[i]] = first_label[i]
    return relation_dict


@log_time_delta
def load_data_first():
    """
    :param data_folder_path: str
    :return: (list, list, list, list)
    
    Only for 粗分类（4） Task
    """
    # 634 -> 4
    relation_dict = load_relations_first()

    # 634
    with codecs.open(config.train_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        qs = [0] * len(all_lines)
        qs_class = [0] * len(all_lines)
        for i, line in enumerate(all_lines):
            qs[i] = line.strip().split("\t")[0]
            qs_class[i] = line.strip().split("\t")[1]
        del all_lines

    with codecs.open(config.test_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        qs_test = [0] * len(all_lines)
        qs_class_test = [0] * len(all_lines)
        for i, line in enumerate(all_lines):
            qs_test[i] = line.strip().split("\t")[0]
            qs_class_test[i] = line.strip().split("\t")[1]
        del all_lines

    # -> 4
    label = [0] * len(qs)
    label_test = [0] * len(qs_test)
    for i, v in enumerate(qs_class):
        try:
            label[i] = relation_dict[v]
        except KeyError as e:
            print("We have encountered a KeyError", e)
            label[i] = "UNK"

            # remove the outlier
            del qs[label.index("UNK")]
            del label[label.index("UNK")]
            print("We have remove the missing question label.\n")
    print("Training data has been successfully loaded.")
    
    for i, v in enumerate(qs_class_test):
        try:
            label_test[i] = relation_dict[v]
        except KeyError as e:
            print("We have encountered a KeyError", e)
            label_test[i] = "UNK"

            # remove the outlier
            del qs_test[label_test.index("UNK")]
            del label_test[label_test.index("UNK")]
            print("We have remove the missing question label.\n")
    print("Testing data has been successfully loaded.")

    if not os.path.exists(config.hash_4_folder):
        os.mkdir(config.hash_4_folder)
    # "qs", "label", "qs_test", "label_test"
    logger.info("Writing the data...")
    for n in [config.ini["data"]["qs"], config.ini["data"]["label"], config.ini["data"]["qs_test"], config.ini["data"]["label_test"]]:
        logger.info("dataset pkl: %s" % "{}/{}".format(config.hash_4_folder, n))
        with codecs.open("{}/{}".format(config.hash_4_folder, n), "wb") as f:
            pickle.dump(locals()[n.split(".")[0]], f)
    del qs, label, qs_test, label_test


@log_time_delta
def load_data_second():
    """
    
    Only for 分类（42） Task
    """
    # 634 -> 42
    relation_dict = load_relations_second()

    # 634
    with codecs.open(config.train_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        qs = [0] * len(all_lines)
        qs_class = [0] * len(all_lines)
        for i, line in enumerate(all_lines):
            qs[i] = line.strip().split("\t")[0]
            qs_class[i] = line.strip().split("\t")[1]
        del all_lines

    with codecs.open(config.test_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        qs_test = [0] * len(all_lines)
        qs_class_test = [0] * len(all_lines)
        for i, line in enumerate(all_lines):
            qs_test[i] = line.strip().split("\t")[0]
            qs_class_test[i] = line.strip().split("\t")[1]
        del all_lines

    # -> 42
    label = [0] * len(qs)
    label_test = [0] * len(qs_test)
    for i, v in enumerate(qs_class):
        try:
            label[i] = relation_dict[v]
        except KeyError as e:
            print("We have encountered a KeyError", e)
            label[i] = "UNK"

            # remove the outlier
            del qs[label.index("UNK")]
            del label[label.index("UNK")]
            print("We have remove the missing question label.\n")
    print("Training data has been successfully loaded.")
    
    for i, v in enumerate(qs_class_test):
        try:
            label_test[i] = relation_dict[v]
        except KeyError as e:
            print("We have encountered a KeyError", e)
            label_test[i] = "UNK"

            # remove the outlier
            del qs_test[label_test.index("UNK")]
            del label_test[label_test.index("UNK")]
            print("We have remove the missing question label.")
    print("Testing data has been successfully loaded.")
    if not os.path.exists(config.hash_42_folder):
        os.mkdir(config.hash_42_folder)
    logger.info("Writing the data...")
    for n in [config.ini["data"]["qs"], config.ini["data"]["label"], config.ini["data"]["qs_test"], config.ini["data"]["label_test"]]:
        logger.info("dataset pkl: %s" % "{}/{}".format(config.hash_42_folder, n))
        with codecs.open("{}/{}".format(config.hash_42_folder, n), "wb") as f:
            pickle.dump(locals()[n.split(".")[0]], f)
    del qs, label, qs_test, label_test


@log_time_delta
def load_data_third():
    """Load Dataset for 633 Classification
    
    [description]
    
    Decorators:
        log_time_delta
    """
    logger.info("Start loading data...")
    with codecs.open(config.train_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        qs = [0] * len(all_lines)
        label = [0] * len(all_lines)
        for i, line in enumerate(all_lines):
            qs[i] = line.strip().split("\t")[0]
            label[i] = line.strip().split("\t")[1]
        del all_lines

    with codecs.open(config.test_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
        qs_test = [0] * len(all_lines)
        label_test = [0] * len(all_lines)
        for i, line in enumerate(all_lines):
            qs_test[i] = line.strip().split("\t")[0]
            label_test[i] = line.strip().split("\t")[1]
        del all_lines
    logger.info("Finished loading data...")

    if not os.path.exists(config.hash_633_folder):
        os.mkdir(config.hash_633_folder)

    for n in [config.ini["data"]["qs"], config.ini["data"]["label"], config.ini["data"]["qs_test"], config.ini["data"]["label_test"]]:
        logger.info("dataset pkl: %s" % "{}/{}".format(config.hash_633_folder, n))
        with codecs.open("{}/{}".format(config.hash_633_folder, n), "wb") as f:
            pickle.dump(locals()[n.split(".")[0]], f)
    logger.info("Pickled all the data...")
    del qs, label, qs_test, label_test


def main():
    load_data_first()
    load_data_second()
    load_data_third()


if __name__ == "__main__":
    main()
