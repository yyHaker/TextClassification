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
from myutils.nn import LSTM, Linear


class TextRNN(nn.Module):
    """Text RNN model"""
    def __init__(self, config, pretrained_emb):
        super(TextRNN, self).__init__()

        self.config = config["arch"]["args"]

        # word embedding layer
        self.word_emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)

        # RNN
        self.rnn = LSTM(
            input_size=self.config["word_dim"],
            hidden_size=self.config["hidden_size"],
            batch_first=True,
            num_layers=1,
            bidirectional=False,
            dropout=self.config["drop_out"]
        )

        # fully-connected layers
        self.fc = Linear(self.config["hidden_size"], self.config["n_classes"])

    def forward(self, data):
        text, length = data  # (b, seq_len), (b)
        batch_size = text.size()[0]
        x = self.word_emb(text)  # (b, seq_len, d)
        x, _ = self.rnn(x)  # (b, seq_len, d)
        x = x[:, -1, :].view(batch_size, -1)  # (b, d)
        x = self.fc(x)
        _, pred = torch.max(x, dim=-1)
        return x, pred
