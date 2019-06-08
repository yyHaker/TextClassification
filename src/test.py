#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: test.py.py
@time: 2019/3/9 15:21
"""
import argparse
import json
import os
import logging

import torch

import data_loader as module_data
import model as module_arch
from myutils import ensure_dir
import codecs
import csv


def predict(args):
    """
    use the best model to test...
    :param args:
    :return:
    """
    # get logger
    logger = logging.getLogger('TC')

    # load best model and params
    model_path = os.path.join(args.path, args.model)
    state = torch.load(model_path)
    config = state["config"]  # test file path is in config.json

    logger.info('Best result on dev is {}'.format(state['monitor_best']))
    config['data_loader']['args']['dev_batch_size'] = args.batch_size
    state_dict = state["state_dict"]

    # set test_file
    if not args.test_file:
        raise AssertionError('You should spacify the test file name (like .test1.json)')
    else:
        config['data_loader']['args']['test_file'] = args.test_file

    logger.info('Run test  with config:')
    logger.info(json.dumps(config, indent=True))

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(config)

    device = config["data_loader"]["args"]["device"]

    # build model architecture
    model = getattr(module_arch, config['arch']['type'])(config, data_loader.vocab)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("begin predict examples...")
    preds = []
    with torch.no_grad():
        data_iter = data_loader.test_iter
        for batch_idx, data in enumerate(data_iter):
            ids, input_datas, labels = data.ID, data.review, data.label
            output, pred_idxs = model(input_datas)
            positive_prob = output[:, 1]
            # get "positive" prob list
            for i in range(output.size()[0]):
                pred = []
                pred += ids[i]
                pred += positive_prob[i].item()
                preds.append(pred)
            if batch_idx % 10 == 0:
                logger.info("predict {} samples done!".format((batch_idx + 1) * output.size()[0]))

    logger.info("write result to file....")
    predict_file = args.target
    ensure_dir(os.path.split(predict_file)[0])
    with codecs.open(predict_file, 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        for pred in preds:
            writer.writerow(["ID", "Pred"])
            writer.writerow(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='batch_size')
    parser.add_argument('-p', "--path", default="./result/2019kesci/saved", type=str, help="best model directory")
    parser.add_argument('-m', '--model', default=None, type=str, help="best model name(.pth)")
    parser.add_argument('-t', "--target", default="./result/predict/result.csv", type=str,
                        help="prediction result file")
    parser.add_argument("--test_file", default="data/2019kesci/20190520_test.csv", type=str,
                        help="prediction result file")
    parser.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    parser.add_argument('--on_dev', default=False, action='store_true', help='Whether get pred_result on dev')
    args = parser.parse_args()

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # prpare logger
    logger = logging.getLogger('TC')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    predict(args)
