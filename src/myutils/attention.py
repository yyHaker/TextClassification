#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: attention.py
@time: 2019/5/22 09:48
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Self-attention"""
    def __init__(self, input_dim, hidden_dim=64):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_outputs, mask=None):
        """
        :param encoder_outputs: (BxLxD)
        :param mask: (BXL), like:
                  [[1, 1, 1, -inf, -inf],
                   [1, 1, 1, 1,-inf],
                   [1, 1, 1, 1, 1 ]]
        :return: outputs, (BxD)
                      weight, (BxL)
        """
        # (BxLxD) -> (BxLx1) -> (BxL)
        energy = self.projection(encoder_outputs).squeeze(-1)
        if mask is not None:
            energy = energy * mask
        # weight: (BxL)
        weight = F.softmax(energy, dim=-1)
        # (BxLxD) * (BxLx1) -> (BxD)
        outputs = (encoder_outputs * weight.unsqueeze(-1)).sum(dim=1)
        return outputs, weight
