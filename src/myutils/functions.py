#!/usr/bin/python
# coding:utf-8

"""some useful functions for build models.
@author: yyhaker
@contact: 572176750@qq.com
@file: functions.py
@time: 2019/5/20 22:36
"""
import torch
INF = 1e30


def masked_softmax(x, m=None, dim=-1):
    """
    Softmax with mask.
    :param x: the Tensor to be softmaxed.
    :param m: mask.
    :param dim:
    :return:
    """
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax


def seq_mask(seq_len, device, max_len=None, inf=False):
    '''
    mask a seq.
    :param seq_len: [b]
    :param device:
    :param max_len:
    :return: mask matrix
    '''
    batch_size = seq_len.size(0)
    if not max_len:
        max_len = torch.max(seq_len)
    mask = torch.zeros((batch_size, max_len), device=device)
    if inf:
        mask = torch.ones((batch_size, max_len), device=device) * (-INF)
    for i in range(batch_size):
        for j in range(seq_len[i]):
            mask[i][j] = 1
    return mask


def softmax_mask(A, mask, dim=1, epsilon=1e-12):
    '''
        applay oftmax on A and consider mask
        :param A:
        :param mask:
        :param dim:
        :param epsilon:
        :return:
        '''
    # According to https://discuss.pytorch.org/t/apply-mask-softmax/14212/7
    A_max = torch.max(A, dim=dim, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * mask  # this step masks
    A_softmax = A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + epsilon)
    return A_softmax


def log_softmax_mask(A, mask, dim=1, epsilon=1e-12):
    '''
    applay log_softmax on A and consider mask
    :param A:
    :param mask:
    :param dim:
    :param epsilon:
    :return:
    '''
    # According to https://discuss.pytorch.org/t/apply-mask-softmax/14212/7
    return torch.log(softmax_mask(A, mask, dim=1, epsilon=epsilon))