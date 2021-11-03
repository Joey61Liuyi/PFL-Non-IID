# -*- coding: utf-8 -*-
# @Time    : 2021/10/26 15:03
# @Author  : LIU YI

import torch
from flcore.optimizers.fedoptimizer import CosineAnnealingLR

model = torch.nn.Linear(10,10)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.005)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scheduler = CosineAnnealingLR(optimizer, 2,95, 95, 0.0)
plot_list = []

import matplotlib.pyplot as plt

for i in range(110):
    scheduler.update(i, 0.0)
    # scheduler.step()
    for j in range(100):
        scheduler.update(None, j/100)

        plot_list.append(scheduler.get_lr())

plt.plot(plot_list)
print(plot_list)
plt.show()

