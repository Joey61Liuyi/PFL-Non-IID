import sys

import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from flcore.optimizers.fedoptimizer import CosineAnnealingLR, OrCosineAnnealingLR
import numpy as np
import time


class clientFedMD(Client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size,
                 learning_rate,
                 local_steps):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size,
                         learning_rate,
                         local_steps)

        self.loss = nn.CrossEntropyLoss()
        self.MD_loss = nn.KLDivLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.MD_optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005)
        self.aggregated_logits = None
        self.MD_logits = None
        # self.scheduler = OrCosineAnnealingLR(self.optimizer, 5, 100, 95, 1e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.65)

    def predict(self, x):
        self.model.eval()
        # logits = self.nas_competetive_output(self.model(x))
        output = self.model(x)
        if isinstance(output, tuple):
            logits, cell_result = output
            for j in range(len(cell_result)):
                cell_result[j] = cell_result[j].detach().view(-1)
                cell_result[j] = F.log_softmax(cell_result[j])
            cell_result.append(logits.detach())
        # self.MD_logits = cell_result
        return cell_result

    def MD_aggregation(self,x,  aggregated_logits):
        self.model.train()
        self.MD_optimizer.zero_grad()
        output = self.model(x)
        if isinstance(output, tuple):
            logits, cell_result = output
            for j in range(len(cell_result)):
                cell_result[j] = cell_result[j].view(-1)
                cell_result[j] = F.log_softmax(cell_result[j])
            cell_result.append(logits)

        loss = 0
        for i in range(len(cell_result)):
            loss += self.MD_loss(cell_result[i], aggregated_logits[i])/len(cell_result)
        # loss = [self.MD_loss(cell_result[i], aggregated_logits[i]) for i in range(len(output))]
        # output = self.nas_competetive_output(output)
        # loss = self.MD_loss(output[0], aggregated_logits[0]) + self.MD_loss(output[1], aggregated_logits[1])
        # loss = torch.sum(loss_list)
        loss.backward()
        self.MD_optimizer.step()

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
            # output = output[1]
            if torch.any(torch.isnan(output)):
                sys.exit()
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        # self.model.cpu()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time