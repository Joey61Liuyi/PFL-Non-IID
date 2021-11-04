# -*- coding: utf-8 -*-
# @Time    : 2021/10/26 15:03
# @Author  : LIU YI

import torch
from flcore.optimizers.fedoptimizer import CosineAnnealingLR, OrCosineAnnealingLR

model = torch.nn.Linear(10,10)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.025)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# scheduler = CosineAnnealingLR(optimizer, 5, 100, 95, 0.0)
scheduler = OrCosineAnnealingLR(optimizer, 5, 100, 95, 6.408e-5)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40, 60, 80, 90], gamma=0.65)

plot_list = []

import matplotlib.pyplot as plt

for i in range(120):
    scheduler.update(i, 0.0)
    # scheduler.step(i)
    for j in range(100):
        scheduler.update(None, j/100)
        plot_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

plt.plot(plot_list)
print(plot_list)
plt.show()

