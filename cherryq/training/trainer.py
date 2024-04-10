from torch.optim.lr_scheduler import LambdaLR

from .base.fsdp_trainer import FSDPTrainer


class Trainer(FSDPTrainer):
    def __init__(self, *args, min_warmup_ratio=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.min_warmup_ratio = min_warmup_ratio

    def create_scheduler(self, num_training_steps: int, optimizer=None): 
        scheduler = super().create_scheduler(num_training_steps, optimizer)
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)

        old_get_lr = scheduler.get_lr
        
        def get_lr(scheduler_self):
            if not isinstance(scheduler_self, LambdaLR):
                return old_get_lr()
            
            lrs = []
            for lmbda, base_lr in zip(scheduler_self.lr_lambdas, scheduler_self.base_lrs):
                if scheduler_self.last_epoch < num_warmup_steps:
                    lrs.append(base_lr * lmbda(scheduler_self.last_epoch))
                else:
                    r = lmbda(scheduler_self.last_epoch)
                    r = 1 - (1 - r) * (1 - self.min_warmup_ratio)
                    lrs.append(base_lr * r)
            return lrs
            
        scheduler.get_lr = get_lr.__get__(scheduler, type(scheduler))
        return scheduler
    