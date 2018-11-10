# -*- coding: utf-8 -*-
# @Author: Yue Peng
# @Email: yuepaang@gmail.com
# @Date: Nov 05, 2018
# @Created on: 22:02:18
# @Description: Lock Dropout
import torch
import torch.nn as nn
from torch.autograd import Variable


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()
    
    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


def main():
    model = LockedDropout()
    x = torch.ones(32, 40, 300)
    print(model(x))


if __name__ == '__main__':
    main()
