def adjust_learning_rate(optimizer, lr):
    print('\033[1;36m' + 'Adjust Learning Rate => %.4e' % lr + '\033[0m')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MultiStepSetLR:
    def __init__(self, optimizer, schedule, epoch=0) -> None:
        self.optimizer = optimizer
        self.schedule = schedule
        self.epoch = epoch

    def step(self):
        self.epoch += 1
        if self.epoch in self.schedule.keys():
            adjust_learning_rate(self.optimizer, self.schedule[self.epoch])

    def state_dict(self):
        return {'epoch': self.epoch}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
