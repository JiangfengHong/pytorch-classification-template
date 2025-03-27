import os 
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,random_split
from config import Config, Logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.util import get_optimizer, initialize_weights, get_criterion, prepare_device
from torch.optim.lr_scheduler import LambdaLR
from data.dataset import TFs_Dataset
import random

from model.mymodel import *
from model.AE import AutoEncoder
from callbacks.lrscheduler import StepLr,ExpDecayLr
from callbacks.modelcheckpoint import ModelCheckpoint
from callbacks.writetensorboard import WriterTensorboardX
from callbacks.earlystopping import EarlyStopping
from trainer.trainer import Trainer

st = time.strftime('%Y%m%d_%H%M', time.localtime())

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()

# dataset paramater
parser.add_argument('--data_path', type=str, default='/media/star/数据集/hongjf/code/old/UAV_OSR_V3_0/Data', help='path to dataset')
parser.add_argument('--select_class', type=str, default='1,2,3', help='Select class (comma-separated list, e.g., "1,2,3" or "all")')
parser.add_argument('--len_time', type=int, default=1, help='time steps')
parser.add_argument('--size', type=int, default=512, help='image size')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used in dataloading')
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--is_test', type=bool, default=False)


# train parameter
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_path', type=str, default='/media/star/数据集/hongjf/code/new/pytorch-template-main/output/20250326_1823/checkpoint_dir/ORACLE-model_best_epoch_88.pth')


# model
# parser.add_argument('--arch', type=str, default='ORACLE')
# parser.add_argument('--num_classes', type=int, default='24', help='Number of classes')
# parser.add_argument('--in_channels', type=int, default='1', help='Number of input channels')

# AE
parser.add_argument('--arch', type=str, default='AutoEncoder')
parser.add_argument('--feat_num',type=str,default='32',help='dim of feature')
parser.add_argument('--in_channels', type=int, default='1', help='Number of input channels')

# Loss function and Optimizer parameter
parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--criterion', choices=['CrossEntropyLoss'], default='CrossEntropyLoss', help='Loss function: CrossEntropyLoss')

#---------------------------------callbacks-------------------------------------
# earlystopping
parser.add_argument('--mode', type=str, default='min', help='monitoring mode')
parser.add_argument('--patience', type=int, default=16)
parser.add_argument('--verbose', type=bool, default=True)

# lr_scheduler
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay rate.')
parser.add_argument('--lr_decay_epoch', type=int, default=5)
parser.add_argument('--is_stair', type=bool, default=True)

# modelcheckpoint
parser.add_argument('--monitor', type=str, default='val_loss', help='Quantity to monitor.')
parser.add_argument('--save_best_only',type=bool, default=True, help='Save the model when it is the best.')
parser.add_argument('--save_epochs', type=int, default=5, help='Save model checkpoints every k epochs.')


# other
parser.add_argument('--N_GPU', type=int, default='2',help='Number of GPUs to use.')
parser.add_argument('--config_file', type=str, default='./config.json')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--topk', type=int, default=1)


# 文件保存路径
parser.add_argument('--log_dir', type=str, default=f'./output/{st}/logs/')
parser.add_argument('--checkpoint_dir', type=str, default=f'./output/{st}/checkpoint_dir')
parser.add_argument('--tensorboard_dir',type=str,default=f'./output/{st}/TSboard')

# 解析参数
args = parser.parse_args()

# 如果 select_class 是字符串，将其转换为列表
if args.select_class is not None:
    if args.select_class.lower() == 'all':
        args.select_class = None  # 选择None 作为特殊值
        args.num_classes = 24
    else:
        args.select_class = [int(item) for item in args.select_class.split(',')]  # 转换为整数列表
        args.num_classes = len(args.select_class)

logger = Logger(log_dir=args.log_dir)

cfg = Config(logger=logger, args=args)
cfg.print_config()
cfg.save_config(cfg.config['config_file'])

# 设定随机种子
seed_torch(cfg.config['seed'])

# dataloader
total_dataset = TFs_Dataset(data_path=cfg.config['data_path'], 
                            logger=logger, 
                            select_class=cfg.config['select_class'], 
                            len_time=cfg.config['len_time'], 
                            size=cfg.config['size']
                            )

if cfg.config['is_test']:
    total_size = len(total_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(total_dataset, [train_size,val_size,test_size])
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

else:
    total_size = len(total_dataset)
    train_size = int(0.7 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.config['batch_size'], shuffle=False)
    test_loader = None


train_step_size = len(train_loader)

# model 
# model = res_embedding_4_layers(n_classes=cfg.config['num_classes'],in_channel=1)
model = AutoEncoder(feat_dim=cfg.config['feat_dim'],in_channel=cfg.config['in_channels'],size=cfg.config['size'])
logger.info(model)

# optimizer and criterion
param = [p for p in model.parameters() if p.requires_grad]
optimizer = get_optimizer(cfg.config['optimizer'], param, lr=cfg.config['lr'])

criterion = get_criterion(cfg.config['criterion'])


# callbacks
logger.info("initializing callbacks")

early_stop = EarlyStopping(mode=cfg.config['mode'],
                        patience=cfg.config['patience'],
                        verbose=cfg.config['verbose'])

lr_scheduler = ExpDecayLr(optimizer=optimizer,lr = cfg.config['lr'],
                        decay_rate=cfg.config['lr_decay'],
                        total_steps=train_step_size * cfg.config['epochs'],
                        steps_per_epoch=train_step_size, 
                        decay_epochs=cfg.config['lr_decay_epoch'],
                        is_stair=cfg.config['is_stair'])

model_checkpoint = ModelCheckpoint(checkpoint_dir=cfg.config['checkpoint_dir'],
                        monitor=cfg.config['monitor'],
                        logger=logger,
                        save_best_only=cfg.config['save_best_only'],
                        mode=cfg.config['mode'],
                        epoch_freq=cfg.config['save_epochs'],
                        arch=cfg.config['arch'])

writer_summary = WriterTensorboardX(writer_dir=cfg.config['tensorboard_dir'], 
                        logger=logger, 
                        enable=True)

# trainer
logger.info('training model....')
model.apply(initialize_weights)
trainer = Trainer(model=model,
                optimizer=optimizer,
                criterion=criterion,
                cfg=cfg,
                logger=logger,
                train_loader=train_loader,
                val_loader=val_loader,
                lr_scheduler=lr_scheduler,
                writer=writer_summary,
                early_stopping=early_stop,
                model_checkpoint=model_checkpoint
                )

trainer.train()
# trainer.test()