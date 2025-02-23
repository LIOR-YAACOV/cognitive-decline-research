import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler


class LearningRateWarmUP(_LRScheduler):
    def __init__(self, optimizer, warm_up_epochs, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warm_up_epochs = warm_up_epochs
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.current_epoch = 1
        self.warmup_learning_rate()

    def warmup_learning_rate(self):
        warmup_lr = self.target_lr * float(self.current_epoch) / float(self.warm_up_epochs)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, loss_value = None):
        if self.current_epoch <= self.warm_up_epochs:
            self.warmup_learning_rate()
        else:
            self.after_scheduler.step(loss_value)
        
        self.current_epoch = self.current_epoch + 1
  
    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)


if __name__ == '__main__':
    v = torch.zeros(10)
    lr = 1e-2
    total_iter = 100
    warmup_iter = 10

    optim = torch.optim.SGD([v], lr=lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_iter - warmup_iter)
    scheduler = LearningRateWarmUP(optimizer=optim,
                                   warm_up_epochs=warmup_iter,
                                   target_lr=lr,
                                   after_scheduler=scheduler_cosine)

    x_iter = [0]
    y_lr = [0.]

    for iter in range(1, total_iter + 1):
        print("iter: ", iter, " ,lr: ", optim.param_groups[0]['lr'])

        optim.zero_grad()
        optim.step()

        scheduler.step()

        x_iter.append(iter)
        y_lr.append(optim.param_groups[0]['lr'])

    plt.plot(x_iter, y_lr, 'b')
    plt.legend(['learning rate'])
    plt.xlabel('iteration')
    plt.ylabel('learning rate')
    plt.show()
