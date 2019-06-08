#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: trainer.py
@time: 2019/3/9 15:45
"""
import torch
import numpy as np
import fnmatch
from .base_trainer import BaseTrainer
from myutils import *


class Trainer(BaseTrainer):
    """
    Trainer class.
    Note:
        Inherited from BaseTrainer.
        ------
        realize the _train_epoch method.
    """
    def __init__(self, model, loss, metrics, optimizer, resume, config,
                 data_loader):
        """Trainer.
        :param model:
        :param loss:
        :param metrics:
        :param optimizer:
        :param resume:
        :param config:
        :param data_loader:
        :param logger:
        """
        super(Trainer, self).__init__(model, loss, metrics, optimizer, resume, config)
        # data loader
        self.data_loader = data_loader
        # if do validation
        self.do_validation = self.data_loader.valid_iter is not None

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch.

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        self.model.train()
        total_loss = 0.
        total_acc_num = 0.
        # begin train
        self.data_loader.train_iter.device = self.device
        for batch_idx, data in enumerate(self.data_loader.train_iter):
            # build data
            text, label = data.text, data.label
            # forward and backward
            self.optimizer.zero_grad()
            output, pred = self.model(text)
            loss = self.loss(output, label)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * output.size()[0]

            acc = self._accuracy(pred, label)
            total_acc_num += acc * pred.size()[0]

            # log for several batches
            # if batch_idx % self.log_step == 0:
            #     self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)], Loss: {:.6f}, ACC: {:.6f}'.format(
            #         epoch,
            #         batch_idx,
            #         len(self.data_loader.train_iter),
            #         100.0 * batch_idx / (len(self.data_loader.train_iter)),
            #         loss.item(), acc))

            # add scalar to writer
            global_step = (epoch - 1) * len(self.data_loader.train_iter) + batch_idx
            self.writer.add_scalar('train loss', loss.item(), global_step=global_step)

        # if train
        avg_loss = total_loss / (len(self.data_loader.train) + 0.)
        avg_acc = total_acc_num / len(self.data_loader.train)
        metrics = np.array([avg_loss])
        result = {
            "train_metrics": metrics
        }
        # if evaluate
        if self.do_validation:
            result = self._valid_epoch(epoch)
        self.logger.info("Training epoch {} done, avg loss: {}, train avg acc: {}, valid {}: {}".format(epoch,
                                                                                                        avg_loss, avg_acc, self.monitor, result[self.monitor]))
        # add to writer
        self.writer.add_scalar("valid_" + self.monitor, result[self.monitor], global_step=epoch * len(self.data_loader.train))
        return result

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        total_loss = 0.
        total_acc_num = 0.
        with torch.no_grad():
            self.data_loader.valid_iter.device = self.device
            for batch_idx, data in enumerate(self.data_loader.valid_iter):
                text, label = data.text, data.label
                output, pred = self.model(text)
                loss = self.loss(output, label)

                acc = self._accuracy(pred, label)

                total_loss += loss.item() * output.size()[0]
                total_acc_num += acc * pred.size()[0]

                # add scalar to writer
                global_step = (epoch - 1) * len(self.data_loader.valid_iter) + batch_idx
                self.writer.add_scalar('eval loss', loss.item(), global_step=global_step)

        # evaluate
        val_loss = total_loss / len(self.data_loader.valid)
        # TODO: calc monitor value
        val_acc = total_acc_num / len(self.data_loader.valid)

        self.logger.info('Valid Epoch: {}, loss: {:.6f}'.format(epoch, val_loss))
        # metrics dict
        metrics = {}
        metrics[self.monitor] = val_acc
        return metrics

    def _accuracy(self, pred, target):
        return float(torch.sum(torch.eq(pred, target))) / (pred.size()[0] + 0.0)

    def test(self):
        """After train done, use the best model to test."""
        # load best model
        best_model_path = self.checkpoint_dir
        for file in os.listdir(self.checkpoint_dir):
            if fnmatch.fnmatch(file, 'model_best_{}_*.pth'.format(self.monitor)):
                best_model_path = os.path.join(self.checkpoint_dir, file)
        state = torch.load(best_model_path)
        state_dict = state["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("begin predict examples...")
        total_acc_num = 0.
        with torch.no_grad():
            self.data_loader.test_iter.device = self.device
            for batch_idx, data in enumerate(self.data_loader.test_iter):
                text, label = data.text, data.label
                output, pred = self.model(text)

                acc = self._accuracy(pred, label)
                total_acc_num += acc * pred.size()[0]
        # calc test acc
        test_acc = total_acc_num / len(self.data_loader.test)
        self.logger.info("Test done, the best model on test data acc: {:.6f}.".format(test_acc))
