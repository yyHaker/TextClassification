#!/usr/bin/python
# coding:utf-8

"""
@author: Mingxiang Tuo
@contact: tuomx@qq.com
@file: train_w2v.py
@time: 2019/4/21 16:10
"""

import gensim, logging
import json
import argparse
import csv


class MySentences(object):
    def __init__(self, paths):
        self.paths = paths

    def __iter__(self):
        for path in self.paths:
            with open(path, 'r', encoding='utf-8') as fin:
                reader = csv.reader(fin)
                for line in reader:
                    yield line[1].split()


def parse_args():
    parser = argparse.ArgumentParser('Train Word2Vec.')
    parser.add_argument('--min_count', type=int, default=16)
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--train_files', nargs='+', default=['../data/2019kesci/train.csv', '../data/2019kesci/valid.csv', '../data/2019kesci/20190520_test.csv'],
                        help='list of files that contain the train data')
    parser.add_argument('--save_name', type=str, default='train_on_review.100.w2v')
    return parser.parse_args()


def train_word2vec(args):
    sentences = MySentences(args.train_files)
    model = gensim.models.Word2Vec(sentences, size=args.size, min_count=args.min_count, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.save_name)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('Run with args: {}'.format(args))

    train_word2vec(args)