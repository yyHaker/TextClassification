#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: mr_data.py
@time: 2019/6/8 22:16
"""
import os
import logging
import json
import torchtext.vocab as vocab
from torchtext.data import Field, LabelField, TabularDataset, BucketIterator
from sklearn.utils import shuffle
from myutils import *


class MR(object):
    """MR data loader."""
    def __init__(self, config):
        # logger
        self.logger = logging.getLogger(config["name"])

        # data loader params
        self.config = config["data_loader"]["args"]

        data_path = self.config["data_path"]
        self.data_path = data_path
        ensure_dir(data_path)
        self.train_path = os.path.join(data_path, self.config["train_file"])
        self.valid_path = os.path.join(data_path, self.config["valid_file"])
        self.test_path = os.path.join(data_path, self.config["test_file"])

        # limit max text length
        self.context_threshold = self.config["context_threshold"]

        self.logger.info("preprocessing data files...")
        if not os.path.exists(self.train_path) or not os.path.exists(self.valid_path) or not os.path.exists(self.test_path):
            self.preprocess()

        # define filed
        TEXT = Field(sequential=True, use_vocab=True, tokenize=lambda x: x, lower=True,
                     include_lengths=True, batch_first=True)
        LABLE = LabelField(sequential=False, use_vocab=False)


        # build dataset
        self.logger.info("building dataset......")

        train_dict_fileds = {
            'text': ('text', TEXT),
            'label': ('label', LABLE)
        }

        self.train, self.valid, self.test = TabularDataset.splits(
            path=data_path,  # data root path
            format="json",
            train=self.config["train_file"],
            validation=self.config["valid_file"],
            test=self.config["test_file"],
            fields=train_dict_fileds
        )

        # build vocab
        self.logger.info("building vocab....")
        TEXT.build_vocab(self.train, self.valid, self.test)

        # load pretrained embeddings
        self.logger.info("load pretrained embeddings...")
        Vectors = vocab.Vectors(self.config["pretrain_emd_file"])
        TEXT.vocab.load_vectors(Vectors)
        # just for call easy
        self.vocab = TEXT.vocab

        # build iterators
        self.logger.info("building iterators.....")
        self.train_iter, self.valid_iter = BucketIterator.splits(
            (self.train, self.valid),
            batch_sizes=(self.config["train_batch_size"], self.config["valid_batch_size"]),
            device=self.config["device"],
            sort_key=lambda x: len(x.text),
            sort_within_batch=False
        )
        self.test_iter = BucketIterator(
            self.test,
            batch_size=self.config["test_batch_size"],
            device=self.config["device"],
            sort_key=lambda x: len(x.text),
            sort=False,
            sort_within_batch=False
        )
        self.logger.info("building iterators done!")
        self.logger.info("Total train data set is: {}, valid data set is: {}, test "
                         "data is: {}".format(len(self.train), len(self.valid), len(self.test)))

    def preprocess(self):
        """load train and valid、test data."""
        texts = []
        labels = []
        with open(self.data_path + "rt-polarity.pos", 'r', encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[: -1]
                labels.append(1)
                texts.append(line.split())
        with open(self.data_path + "rt-polarity.neg", 'r', encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[: -1]
                labels.append(0)
                texts.append(line.split())
        # shuffle data
        texts, labels = shuffle(texts, labels)

        valid_idx = len(texts) // 10 * 8
        test_idx = len(texts) // 10 * 9
        # train
        train_texts = texts[: valid_idx]
        train_labels = labels[: valid_idx]
        # valid
        valid_texts = texts[valid_idx: test_idx]
        valid_labels = labels[valid_idx: test_idx]
        # test
        test_texts = texts[test_idx:]
        test_labels = labels[test_idx:]
        self.save_datas(self.train_path, train_texts, train_labels)
        self.save_datas(self.valid_path, valid_texts, valid_labels)
        self.save_datas(self.test_path, test_texts, test_labels)

    def save_datas(self, save_path, texts, labels):
        """save datas to json file."""
        with open(save_path, 'w', encoding="utf-8") as f:
            for text, label in zip(texts, labels):
                if len(text) > self.context_threshold:
                    text = text[: self.context_threshold]
                d = dict([('text', text), ('label', label)])
                json.dump(d, f)
                print('', file=f)
