#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: text_rnn.py
@time: 2019/6/8 15:26
"""
import torch
import torch.nn as nn
from myutils.nn import LSTM, Linear, GRU


class TextRNN(nn.Module):
    """Text RNN model"""
    def __init__(self, config, pretrained_emb):
        super(TextRNN, self).__init__()

        self.config = config["arch"]["args"]

        # word embedding layer
        self.word_emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)

        # RNN
        self.rnn = GRU(
            input_size=self.config["word_dim"],
            hidden_size=self.config["hidden_size"],
            batch_first=True,
            num_layers=1,
            bidirectional=False,
            dropout=self.config["dropout"]
        )

        # fully-connected layers
        self.fc = Linear(self.config["hidden_size"], self.config["n_classes"])

    def forward(self, data):
        text, length = data  # (b, seq_len), (b)
        x = self.word_emb(text)  # (b, seq_len, d)
        _, h = self.rnn((x, length))  # (b, seq_len, d), (b, d)
        x = self.fc(h)
        _, pred = torch.max(x, dim=-1)
        return x, pred
