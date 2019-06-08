#!/usr/bin/python
# coding:utf-8

"""Common evaluation metrics.
@author: yyhaker
@contact: 572176750@qq.com
@file: metric.py
@time: 2019/3/9 15:38
"""
import torch


def ACC(output, target):
    """
    calc accuracy.
    :param output: [b]
    :param target: [b]
    :return:
    """
    return float(torch.sum(torch.eq(output, target))) / (output.size()[0] + 0.0)
