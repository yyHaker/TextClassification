#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: util.py
@time: 2019/3/9 15:43
"""
import json
import os
import spacy
import jieba
import sys


spacy_en = spacy.load('en')


def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text.strip())]


def CN_tokenizer(text):
    return list(jieba.cut(text))


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).

    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.

    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()
    maketrans = str.maketrans
    translate_dict = dict((c, split) for c in filters)
    translate_map = maketrans(translate_dict)
    text = text.translate(translate_map)
    seq = text.split(split)
    return [i for i in seq if i]


def dumpObj(obj, file):
    """
    dump object to file.
    :param obj:
    :param file:
    :return:
    """
    with open(file, 'w') as f:
        json.dump(obj, f)


def loadObj(file):
    """
    load object from file.
    :param file:
    :return:
    """
    with open(file, 'r') as f:
        obj = json.load(f)
    return obj


def ensure_dir(path):
    """
    ensure the dir exists.
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def convert2dict(data, train=True):
    """convert data to dict"""
    return data.text, data.label


if __name__ == "__main__":
    text = "I like playing computer games."
    sent = "I want to watch tv in living room"
    print(tokenizer(text))
    print(tokenizer(sent))

