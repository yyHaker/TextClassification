#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: text_rcnn.py
@time: 2019/6/8 20:14
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from myutils.nn import LSTM, Linear


class TextRCNN(nn.Module):
    """Text RCNN model."""
    def __init__(self, config, pretrained_emb):
        super(TextRCNN, self).__init__()

        self.config = config["arch"]["args"]

        # word embedding layer
        self.word_emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
        # rnn layer
        self.bi_rnn = LSTM(
            input_size=self.config["word_dim"],
            hidden_size=self.config["hidden_size"],
            batch_first=True,
            num_layers=1,
            bidirectional=True,
            dropout=self.config["dropout"]
        )
        # conv layer
        self.conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.config["hidden_size"] * 2 + self.config["word_dim"],
                out_channels=self.config["hidden_size"],
                kernel_size=self.config["filter_size"]
            ),
            # nn.BatchNorm1d(self.config["hidden_size"]),
            nn.ReLU(inplace=True)
        )
        # full-connected layer
        self.fc = Linear(self.config["hidden_size"] * self.config["kmax_pooling"], self.config["n_classes"])

    def kmax_pooling(self, x, dim=2, k=2):
        """k-max pooling"""
        index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
        return x.gather(dim, index)

    def forward(self, data):
        text, length = data  # (b, seq_len), (b)
        x0 = self.word_emb(text)  # (b, seq_len, d)
        x, h = self.bi_rnn((x0, length))  # (b, seq_len, 2*d), (b, 2*d)
        x = torch.cat((x0, x), dim=-1).permute(0, 2, 1)  # (b, seq_len, 3*d)  --> (b, 3*d, seq_len)
        # x = torch.tanh(self.conv(x))   # (b, d, seq_len - filter_size + 1)
        x = self.conv_layer(x)  # (b, d, seq_len - filter_size + 1)
        x = self.kmax_pooling(x, dim=2, k=self.config["kmax_pooling"])
        x = x.reshape(x.size()[0], -1)  # (b, k*d)
        x = self.fc(x)
        _, pred = torch.max(x, dim=-1)
        return x, pred

