import torch
import copy
import time
import numpy as np
import math
from flcore.clients.clientamp import clientAMP, weight_flatten
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import wandb


class FedAMP(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                 num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold, 
                 alphaK, lamda, sigma, run_name):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                         num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold, run_name)
        # select slow clients
        self.set_slow_clients()

        self.alphaK = alphaK
        self.sigma = sigma

        self.client_ws = [model for i in range(num_clients)]
        self.client_us = [model for i in range(num_clients)]

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # train, test = read_client_data(dataset, i)
            client = clientAMP(device, i, train_slow, send_slow, self.train_all[i], self.test_all[i], model, batch_size, learning_rate,
                                local_steps, alphaK, lamda)
            self.clients.append(client)
        del (self.train_all)
        del (self.test_all)

        print(f"\nJoin clients / total clients: {self.join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.start_epoch, self.global_rounds+1):
            print(f"\n-------------Round number: {i}-------------")
            self.send_models()
            if i<self.global_rounds/2:
                eval_gap = 50
            elif i< self.global_rounds*9/10 and i>=self.global_rounds/2:
                eval_gap = 20
            else:
                eval_gap = 1
            if i%eval_gap == 0:
                print("\nEvaluate global model")
                test_acc, train_acc, train_loss, personalized_acc = self.evaluate(i)
                info_dict = {
                    "learning_rate": self.clients[0].optimizer.state_dict()['param_groups'][0]['lr'],
                    "global_valid_top1_acc": test_acc * 100,
                    "average_valid_top1_acc": personalized_acc * 100,
                    "epoch": i
                }
                # print(info_dict)
                wandb.log(info_dict)
            for client in self.clients:
                # client.scheduler.update(i, 0.0)
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.update_client_temp()

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss), personalized_acc)

        self.save_results()
        if i % 100 == 0:
            self.save_global_model_middle(i)


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            if client.send_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))

            client.set_parameters(copy.deepcopy(self.client_us[client.id]))

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                self.client_ws[client.id] = copy.deepcopy(client.model)

    def update_client_temp(self):
        weights = [weight_flatten(mw) for mw in self.client_ws]

        for i, mu in enumerate(self.client_us):
            for param in mu.parameters():
                param.data = torch.zeros_like(param.data)

            coef = torch.zeros(self.num_clients)
            for j, mw in enumerate(self.client_ws):
                if i != j:
                    sub = (weights[i] - weights[j]).view(-1)
                    sub = torch.dot(sub, sub)
                    coef[j] = self.alphaK * self.e(sub)
            coef[i] = 1 - torch.sum(coef)

            for j, mw in enumerate(self.client_ws):
                for param, param_j in zip(mu.parameters(), mw.parameters()):
                    param.data += coef[j] * param_j

    def e(self, x):
        return math.exp(-x/self.sigma)/self.sigma
