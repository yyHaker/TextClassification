#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: model.py
@time: 2019/3/9 15:40
"""
import torch
import torch.nn as nn
from myutils import GRU, SelfAttention, Linear, seq_mask


class RNNAttention(nn.Module):
    """Text RNN Attention model."""
    def __init__(self, config, pretrained_emb):
        super(RNNAttention, self).__init__()

        self.config = config["arch"]["args"]

        # word embedding layer
        self.word_emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=True)

        # lstm layer
        self.rnn = GRU(
            input_size=self.config["word_dim"],
            hidden_size=self.config["hidden_size"],
            batch_first=True,
            bidirectional=True,
            dropout=self.config["dropout"]
        )

        self.self_attention_layer = SelfAttention(self.config["hidden_size"]*2,
                                                  hidden_dim=self.config["hidden_size"]*2)

        # projection layer
        self.projection_layer = Linear(self.config["hidden_size"]*2, self.config["n_classes"])

    def forward(self, data):
        text, length = data  # (b, seq_len), (b)
        text_mask = seq_mask(length, max_len=text.size()[1], device=text.device, inf=True)
        text_emb = self.word_emb(text)  # (b, seq_len, d)
        text_emb, _ = self.rnn((text_emb, length))  # (b, seq_len, hidden*2)

        text_emb, weight = self.self_attention_layer(text_emb, mask=text_mask)  # (b, hidden*2)

        prob = self.projection_layer(text_emb)  # (b, n_classes)
        _, pred = torch.max(prob, dim=-1)
        return prob, pred


