# -*- coding: utf-8 -*-
"""
we want more with our data.

AUTHOR: Yue Peng
EMAIL: yuepeng@sf-express.com
DATE: 2018.10.09
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from cores.utils.baidu_translate import BaiduTranslate
from cores.dataset.data_helper import data_extract

translator = BaiduTranslate()


def chinese_to_foreign(data, label):
    """mapping chinese sentence into foreign language
    
    [description]
    
    Arguments:
        data -- [description]
    """
    data_ext = []
    label_ext = []
    for i, s in enumerate(data):
        for l in translator.FOREIGN_LANG:
            data_ext.append(translator.translate(fromLang="zh", toLang=l, srcString=s))
            label_ext.append(label[i])
    return data_ext, label_ext


def foreign_to_chinese(data, label):
    """mapping foreign language sentence into chinese 
    
    [description]
    
    Arguments:
        data -- [description]
        label -- [description]
    """
    data_ext = []
    label_ext = []
    for i, s in enumerate(data):
        for l in translator.FOREIGN_LANG:
            data_ext.append(translator.translate(fromLang=l, toLang="zh", srcString=s))
            label_ext.append(label[i])
    return data_ext, label_ext


if __name__ == "__main__":
    qs, label, qs_test, label_test, label2id, id2label = data_extract(task="4")
    x, y = chinese_to_foreign(qs, label)
    data_ext, label_ext = foreign_to_chinese(x, y)
