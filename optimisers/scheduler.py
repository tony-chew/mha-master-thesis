from torch.optim.lr_scheduler import _LRScheduler


class PolyLR(_LRScheduler):
    """
    Apply poly learning rate scheduling
    """
    def __init__(self,
                 optimiser,
                 max_batchnum,
                 decay_iter=1,
                 gamma=0.9,
                 last_epoch=-1):
        """
        Initialise the poly learning rate

        :param optimiser: optimiser
        :param max_batchnum: max number of batches for training
        :param decay_iter: chosen interval to decay the learning rate
        :param gamma: gamma parameter
        :last_epoch: the number of the last epoch/batch
        """
        self.decay_iter = decay_iter
        self.max_batchnum = max_batchnum
        self.gamma = gamma
        super(PolyLR, self).__init__(optimiser, last_epoch)

    def get_lr(self):
        """
        Calculate the learning rate for the next iteration based on 'factor'
        """
        factor = (1 - self.last_epoch / float(self.max_batchnum)) ** self.gamma

        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_batchnum:
            base_lr = [base_lr * factor for base_lr in self.base_lrs]
        else:
            base_lr = [base_lr for base_lr in self.base_lrs]

        return base_lr