# -*- coding: utf-8 -*-
"""
Python的单例模式

AUTHOR: Yue Peng
EMAIL: ypeng7@outlook.com
DATE: 2018.10.02
"""
import sys, os
from functools import wraps
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


def singleton(class_):
    instances = {}
    @wraps(class_)
    def get_instance(*args, **kwargs):
        nonlocal instances
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return get_instance


@singleton
class Test(object):
    def __init__(self):
        self.test = "just test!"


def main():
    test = Test()
    print(id(test))
    test_new = Test()
    print(id(test_new))
    print(test.test)


if __name__ == "__main__":
    main()
 