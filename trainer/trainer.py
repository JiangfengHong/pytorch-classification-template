import torch
import numpy as np
from utils.util import ensure_dir, check_dir
from callbacks.progressbar import ProgressBar
from utils.util import prepare_device,AverageMeter
from utils.metrics import Accuracy
import time

class Trainer(object):
    def __init__(self, 
                model, 
                optimizer, 
                criterion, 
                cfg, 
                logger,
                train_loader, 
                val_loader=None, 
                test_loader=None,
                lr_scheduler=None,
                writer=None,
                early_stopping=None,
                model_checkpoint=None
                ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.metric = Accuracy(topK=self.cfg.config['topk'])

        # paramater
        self.batch_num = len(self.train_loader)
        self.epochs = self.cfg.config['epochs']
        self.verbose = self.cfg.config['verbose']
        self.best = np.inf
        self.device, self.device_ids = prepare_device(cfg.config['N_GPU'],logger)

        # callbacks
        self.lr_scheduler = lr_scheduler
        self.writer = writer
        self.progressbar = ProgressBar(n_batch=self.batch_num)
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint

        #******** 单机多GPU情况 ***********
        #整个过程可以这么理解:
        #首先将模型加载到一个指定设备上作为controller,
        # 然后将模型浅复制到多个设备中，将大batch数据也
        #等分到不同设备中， 每个设备分配到不同的数据，然后将所有设备计算得到
        #梯度合并用以更新controller模型的参数。
        if len(self.device_ids) > 1:
            # model = nn.DataParallel(model) 会将模型浅复制到所有可用的显卡中
            # （如果是我实验室的服务器，就是复制到2张卡中）,我们希望占用显卡0和1,
            # 所以需要传入参数device_ids=[0,1]
            self.model = torch.nn.DataParallel(self.model,device_ids=self.device_ids)
        # model = model.cuda() 会将模型加载到0号显卡并作为controller.
        # 但是我们并不打算使用0号显卡。
        # 所以需要修改为：model = model.cuda(device_ids[0]),
        # 即我们将模型加载1号显卡并作为controller
        self.model = self.model.to(self.device)


        # 加载预训练模型
        if self.cfg.config['resume']:
            self._resume_checkpoint(self.cfg.config['resume_path'])
        else:
            self.start_epoch = 1
    
    def save_info(self, epoch):
        state = {
            'arch':self.cfg.config['arch'],
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.cfg,
            'best':self.best
        }
        return state
    
    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self.model)


    def _train_epoch(self, epoch):
        self.model.train()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            start = time.time()

            data = data.to(self.device)
            targets = targets.to(self.device)
            if len(data.shape) != 4:
                data = data.unsqueeze(1)

            outputs = self.model(data)
            loss = self.criterion(outputs, targets.squeeze())
            # 写入TensorBoard，每个batch的损失和准确率
            self.writer.set_step(epoch * self.batch_num + batch_idx)
            self.writer.add_scalar('Batch/loss', loss.item())
            acc = self.metric(outputs,targets)

            # 计算梯度并更新梯度
            #将上次迭代计算的梯度值清0
            self.optimizer.zero_grad()

            # 反向传播，计算梯度值
            # backward只能被应用在一个标量上，也就是一个一维tensor，或者传入跟变量相关的梯度
            loss.backward()
            # 更新权值参数
            self.optimizer.step()
            self.lr_scheduler.step()

            # 更新指标
            # 取得一个tensor的值(返回number), 用.item()
            train_loss.update(loss.item(),data.size(0))
            train_acc.update(acc, data.size(0))
            # 是否打印训练过程
            if self.verbose >= 1:
                self.progressbar.step_classify(epoch = epoch,
                                        batch_idx=batch_idx,
                                        loss  = loss.item(),
                                        acc = acc,
                                        lr = self.optimizer.param_groups[0]['lr'],
                                        use_time = time.time() - start)
        # 写入tensorboard
        self.writer.set_step(epoch)
        self.writer.add_scalar('loss', train_loss.avg)
        self.writer.add_scalar('acc', train_acc.avg)

        # 训练log
        train_log = {
            'loss': train_loss.avg,
            'acc': train_acc.avg
        }
        return train_log
    

    def _valid_epoch(self, epoch):
        # eval()时，pytorch会自动把BN和DropOut固定住,
        # 不会取平均，而是用训练好的值.
        self.model.eval()
        val_losses = AverageMeter()
        val_acc = AverageMeter()
        # 不计算梯度
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.val_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                if len(data.shape) != 4:
                    data = data.unsqueeze(1)
                output = self.model(data)
                loss = self.criterion(output, targets.squeeze())
                acc = self.metric(output,targets)
                val_losses.update(loss.item(),data.size(0))
                val_acc.update(acc.item(),data.size(0))

            # 写入文件中
            self.writer.set_step(epoch, 'valid')
            self.writer.add_scalar('val_loss', val_losses.avg)
            self.writer.add_scalar('val_acc', val_acc.avg)

        return {
            'val_loss': val_losses.avg,
            'val_acc': val_acc.avg
        }


    def train(self):
        for epoch in range(self.start_epoch,self.epochs + 1):
            self.logger.info("\nEpoch {i}/{epochs}......".format(i=epoch, epochs=self.epochs))

            train_log = self._train_epoch(epoch)
            val_log = self._valid_epoch(epoch)
            logs = dict(train_log,**val_log)
            self.logger.info('\nEpoch: %d - loss: %.4f acc: %.4f - val_loss: %.4f - val_acc: %.4f - lr:%.10f'%(
                            epoch,logs['loss'],logs['acc'],logs['val_loss'],logs['val_acc'], self.get_lr()[0])
                             )
            
            # 当满足early_stopping时，停止训练
            if self.early_stopping:
                self.early_stopping.step(epoch=epoch, current=logs[self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break
            # checkpoint
            if self.model_checkpoint:
                # 保存信息
                state = self.save_info(epoch)
                self.model_checkpoint.step(current=logs[self.model_checkpoint.monitor],state = state)


    def test(self):
        pass
        
    def _resume_checkpoint(self, path):
        check_dir(path)
        self.logger.debug('Loading checkpoint: {}...'.format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best = checkpoint['best']
        self.model.load_state_dict(checkpoint['state_dict'])
        if checkpoint['optimizer'] != self.cfg.config['optimizer']:
            self.logger.debug("Optimizer type given in config file is different from that of checkpoint."
                              "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.debug("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def get_lr(self):
        return [param['lr'] for param in self.optimizer.param_groups]