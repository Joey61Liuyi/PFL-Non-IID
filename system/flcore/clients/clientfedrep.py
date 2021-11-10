import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
from flcore.optimizers.fedoptimizer import CosineAnnealingLR, OrCosineAnnealingLR
import numpy as np
import time


class clientFedRep(Client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum = 0.9, weight_decay=0.0005, nesterov=1)
        # self.scheduler = OrCosineAnnealingLR(self.optimizer, 5, 100, 95, 1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.65)
    def train(self):
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        # for step in range(max_local_steps):
        #     if self.train_slow:
        #         time.sleep(0.1 * np.abs(np.random.rand()))
        #
        #     for i, (x, y) in enumerate(self.trainloader):
        #     # x, y = self.get_next_train_batch()
        #         self.scheduler.update(None, 1.0*i/len(self.trainloader))
        #         self.optimizer.zero_grad()
        #         x = x.to(self.device)
        #         y = y.to(self.device)
        #         output = self.model(x)
        #         output = self.nas_competetive_output(output)
        #         loss = self.loss(output, y)
        #         loss.backward()
        #         self.optimizer.step()
        for step in range(max_local_steps):
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            x, y = self.get_next_train_batch()
            self.optimizer.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)
            output = self.model(x)
            output = self.nas_competetive_output(output)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()


        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        temp_dict = model.state_dict()
        target_dict = self.model.state_dict()
        flag_list = list(dict(self.model.named_parameters()).keys())
        # print(flag_list)
        cut_label, flag = (flag_list)[len(flag_list) - 2], False
        for key in flag_list:
            flag = max(flag, (key == cut_label))
            # target_dict[key] = (torch.zeros_like(target_dict[key]) + temp_dict[key]) if flag else target_dict[key]
            target_dict[key] = target_dict[key] if flag else (torch.zeros_like(target_dict[key]) + temp_dict[key])
            # print('flag:',flag)
        self.model.load_state_dict(target_dict)