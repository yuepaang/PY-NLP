# -*- coding: utf-8 -*-
"""
Python的单例模式

AUTHOR: Yue Peng
EMAIL: ypeng7@outlook.com
DATE: 2018.10.02
"""
import sys, os
from functools import wraps


# def singleton(class_):
#     instances = {}
#     @wraps(class_)
#     def get_instance(*args, **kwargs):
#         nonlocal instances
#         if class_ not in instances:
#             instances[class_] = class_(*args, **kwargs)
#         return instances[class_]
#     return get_instance


class Singleton(object):
    def __init__(self, cls):
        wraps(cls)(self)
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            self.instance = self.__wrapped__(*args, **kwargs)
        return self.instance


FORMAT = {
    "ymd": "{d.year}-{d.month}-{d.day}",
    "mdy": "{d.month}/{d.day}/{d.year}"
}


class CustomizedDate(object):
    __slots__ = ["year", "month", "day"]

    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def __format__(self, format_spec):
        if format_spec == "":
            format_spec = "ymd"
        fmt = FORMAT[format_spec]
        return fmt.format(d=self)


# @singleton
@Singleton
class Test(object):
    def __init__(self):
        self.test = "just test!"


def main():
    test = Test()
    print(id(test))
    test_new = Test()
    print(id(test_new))
    print(test.test)

    d = CustomizedDate(2018, 10, 14)
    print(format(d))
    print("Today is {:ymd}".format(d))


if __name__ == "__main__":
    main()
 