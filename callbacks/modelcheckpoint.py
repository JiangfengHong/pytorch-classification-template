import os
import numpy as np
import torch
from utils.util import ensure_dir

class ModelCheckpoint(object):

    def __init__(self, checkpoint_dir,
                 monitor,
                 logger,
                 save_best_only=False,
                 mode='min',
                 epoch_freq=1,
                 arch='ckpt',
                 best = None):
        self.monitor = monitor
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.epoch_freq = epoch_freq
        self.arch = arch
        self.logger = logger
        # 计算模式
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        # 这里主要重新加载模型时候
        #对best重新赋值
        if best:
            self.best = best
        ensure_dir(self.checkpoint_dir)

    def step(self, state, current):
        # 是否保存最好模型
        if self.save_best_only:
            if self.monitor_op(current, self.best):
                self.logger.info('\nEpoch %05d: %s improved from %0.5f to %0.5f'% (state['epoch'], self.monitor, self.best,
                         current))
                self.best = current
                state['best'] = self.best
                best_path = os.path.join(self.checkpoint_dir, '{}-model_best_epoch_{}.pth'.format(state['arch'], state['epoch']))
                torch.save(state, best_path)
                
        else:
            # 每隔epoch_freq保存一次模型
            filename = str(self.checkpoint_dir + '/{}-checkpoint_epoch_{}.pt'.format(state['arch'],current))
            if state['epoch'] % self.epoch_freq == 0:
                self.logger.info("\nEpoch %05d: save model to disk."%(state['epoch']+1))
                torch.save(state, filename)