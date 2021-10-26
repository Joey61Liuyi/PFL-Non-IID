import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
import numpy as np
import time


class clientAVG(Client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)

        self.loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def train(self):
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))

            for i, (x, y) in enumerate(self.trainloader):
            # x, y = self.get_next_train_batch()
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[1]

                if isinstance(output, list):
                    assert len(output) == 2, "output must has {:} items instead of {:}".format(
                        2, len(output)
                    )
                    output, output_aux = output
                else:
                    output, output_aux = output, None

                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
