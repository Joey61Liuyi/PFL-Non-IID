import torch
from torch.optim import Optimizer
import math


class _LRScheduler(object):
    def __init__(self, optimizer, warmup_epochs, epochs):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{:} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        self.base_lrs = list(
            map(lambda group: group["initial_lr"], optimizer.param_groups)
        )
        self.max_epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.current_iter = 0

    def extra_repr(self):
        return ""

    def __repr__(self):
        return "{name}(warmup={warmup_epochs}, max-epoch={max_epochs}, current::epoch={current_epoch}, iter={current_iter:.2f}".format(
            name=self.__class__.__name__, **self.__dict__
        ) + ", {:})".format(
            self.extra_repr()
        )

    def state_dict(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def get_min_info(self):
        lrs = self.get_lr()
        return "#LR=[{:.6f}~{:.6f}] epoch={:03d}, iter={:4.2f}#".format(
            min(lrs), max(lrs), self.current_epoch, self.current_iter
        )

    def get_min_lr(self):
        return min(self.get_lr())

    def update(self, cur_epoch, cur_iter):
        if cur_epoch is not None:
            assert (
                isinstance(cur_epoch, int) and cur_epoch >= 0
            ), "invalid cur-epoch : {:}".format(cur_epoch)
            self.current_epoch = cur_epoch
        if cur_iter is not None:
            assert (
                isinstance(cur_iter, float) and cur_iter >= 0
            ), "invalid cur-iter : {:}".format(cur_iter)
            self.current_iter = cur_iter
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, epochs, T_max, eta_min):
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, warmup_epochs, epochs)

    def extra_repr(self):
        return "type={:}, T-max={:}, eta-min={:}".format(
            "cosine", self.T_max, self.eta_min
        )

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if (
                self.current_epoch >= self.warmup_epochs
                # and self.current_epoch < self.max_epochs
            ):
                last_epoch = self.current_epoch - self.warmup_epochs
                # if last_epoch < self.T_max:
                # if last_epoch < self.max_epochs:
                lr = (
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * last_epoch / self.T_max))
                    / 2
                )
                # else:
                #  lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.T_max-1.0) / self.T_max)) / 2
            # elif self.current_epoch >= self.max_epochs:
            #     lr = self.eta_min
            else:
                lr = (
                    self.current_epoch / self.warmup_epochs
                    + self.current_iter / self.warmup_epochs
                ) * base_lr
            #
            # elif self.current_epoch >= self.max_epochs:
            #     last_epoch = self.max_epochs - self.warmup_epochs -1
            #     lr = (
            #             self.eta_min
            #             + (base_lr - self.eta_min)
            #             * (1 + math.cos(math.pi * last_epoch / self.T_max))
            #             / 2
            #     )
            lrs.append(lr)

        return lrs



class PerAvgOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerAvgOptimizer, self).__init__(params, defaults)

    def step(self, beta=0):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(other=d_p, alpha=-beta)
                else:
                    p.data.add_(other=d_p, alpha=-group['lr'])


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data.add_(- group['lr'] * (p.grad.data + group['eta'] * \
                    self.server_grads[i] - self.pre_grads[i]))
                i += 1


class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1, mu=0.001):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, lamda=lamda, mu=mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)

    def step(self, local_model, device):
        group = None
        weight_update = local_model.copy()
        for group in self.param_groups:
            for p, localweight in zip(group['params'], weight_update):
                localweight = localweight.to(device)
                # approximate local model
                if p.grad != None:
                    p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)
                else:
                    p.data = p.data - group['lr'] * (group['lamda'] * (p.data - localweight.data) + group['mu'] * p.data)

        return group['params']


# class pFedMeOptimizer(Optimizer):
#     def __init__(self, params, lr=0.01, lamda=0.1 , mu = 0.001):
#         #self.local_weight_updated = local_weight # w_i,K
#         if lr < 0.0:
#             raise ValueError("Invalid learning rate: {}".format(lr))
#         defaults = dict(lr=lr, lamda=lamda, mu = mu)
#         super(pFedMeOptimizer, self).__init__(params, defaults)
    
#     def step(self, local_weight_updated, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure
#         weight_update = local_weight_updated.copy()
#         for group in self.param_groups:
#             for p, localweight in zip( group['params'], weight_update):
#                 p.data = p.data - group['lr'] * (p.grad.data + group['lamda'] * (p.data - localweight.data) + group['mu']*p.data)
#         return  group['params'], loss
    
#     def update_param(self, local_weight_updated, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure
#         weight_update = local_weight_updated.copy()
#         for group in self.param_groups:
#             for p, localweight in zip( group['params'], weight_update):
#                 p.data = localweight.data
#         #return  p.data
#         return  group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, beta=1, n_k=1):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')

        default = dict(lr=lr, mu=mu)

        super().__init__(params, default)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])
