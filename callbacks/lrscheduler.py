import math
from torch.optim import Adam

class StepLr(object):
    def __init__(self,optimizer,lr):
        super(StepLr,self).__init__()
        self.optimizer = optimizer
        self.lr = lr

    def step(self,epoch):
        lr = self.lr / (100 * (epoch/4))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

class ExpDecayLr:
    def __init__(self, optimizer, lr, decay_rate, total_steps, steps_per_epoch, decay_epochs, is_stair=False):
        self.optimizer = optimizer
        self.learning_rate = lr
        self.decay_rate = decay_rate
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.decay_epochs = decay_epochs
        self.is_stair = is_stair
        self.current_step = 0

    def step(self):
        if self.is_stair:
            epoch = math.floor(self.current_step / self.steps_per_epoch)
            decayed_steps = math.floor(epoch / self.decay_epochs) * self.decay_epochs
        else:
            decayed_steps = self.current_step / self.steps_per_epoch

        lr = self.learning_rate * math.exp(-self.decay_rate * decayed_steps)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_step += 1
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']



