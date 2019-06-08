#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: text_cnn.py
@time: 2019/5/22 21:53
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """TextCnN for sentence classification"""
    def __init__(self, config, pretrained_emb):
        super(TextCNN, self).__init__()

        self.config = config["arch"]["args"]

        # word embedding layer
        self.word_emb1 = nn.Embedding.from_pretrained(pretrained_emb)
        self.mode = self.config["mode"]
        if self.mode == "static":
            self.word_emb1.weight.requires_grad = False
            self.in_channels = 1
        elif self.mode == "non-static":
            self.word_emb1.weight.requires_grad = True
            self.in_channels = 1
        elif self.mode == "multichannel":
            self.word_emb1.weight.requires_grad = True
            self.word_emb2 = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)
            self.in_channels = 2
        else:
            raise Exception("not supported TextCNN mode.")

        # conv layer
        assert len(self.config["filters"]) == len(self.config["filters_num"])

        for i in range(len(self.config["filters"])):
            conv = nn.Conv1d(in_channels=self.in_channels,
                             out_channels=self.config["filters_num"][i],
                             kernel_size=self.config["word_dim"] * self.config["filters"][i],
                             stride=self.config["word_dim"])
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.config["filters_num"]), self.config["n_classes"])

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, data):
        text, length = data  # (b, seq_len), (b)
        x = self.word_emb1(text).view(-1, 1, self.config["word_dim"] * text.size()[1])  # [b, 1, seq_len*d]
        if self.mode == "multichannel":
            x2 = self.word_emb2(text).view(-1, 1, self.config["word_dim"] * text.size()[1])  # [b, 1, seq_len*d]
            x = torch.cat((x, x2), dim=1)  # (b, 2, seq_len*d)

        # conv(x)  ----->  [b, filter_num, seq_len - filter + 1]
        # max_pool1d ----> [b, filter_num, 1]
        conv_results = [F.max_pool1d(F.relu(self.get_conv(i)(x)), text.size()[1] - self.config["filters"][i] + 1
                         ).view(-1, self.config["filters_num"][i]) for i in range(len(self.config["filters"]))]

        x = torch.cat(conv_results, 1)  # [b, sum(filter_num)]
        x = F.dropout(x, p=self.config["dropout"])
        x = self.fc(x)
        _, pred = torch.max(x, dim=-1)
        return x, pred

