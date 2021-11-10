import wandb
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np

class FedAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                 num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold, run_name):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                         num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold, run_name)
        # select slow clients
        self.set_slow_clients()
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # train, test = read_client_data(dataset, i)
            client = clientAVG(device, i, train_slow, send_slow, self.train_all[i], self.test_all[i], model, batch_size, learning_rate, local_steps)
            self.clients.append(client)
        del(self.train_all)
        del(self.test_all)

        print(f"\nJoin clients / total clients: {self.join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.start_epoch, self.global_rounds+1):
            print(f"\n-------------Round number: {i}-------------")
            self.send_models()
            if i<self.global_rounds/2:
                eval_gap = 50
            elif i< self.global_rounds*95/100 and i>=self.global_rounds/2:
                eval_gap = 20
            else:
                eval_gap = 1
            if i%eval_gap == 0:

                print("\nEvaluate global model")
                test_acc, train_acc, train_loss, personalized_acc = self.evaluate(i)
                info_dict = {
                    "learning_rate": self.clients[0].optimizer.state_dict()['param_groups'][0]['lr'],
                    "global_valid_top1_acc": test_acc*100,
                    "average_valid_top1_acc": personalized_acc*100,
                    "epoch": i
                }
                # print(info_dict)
                wandb.log(info_dict)
            self.selected_clients = self.clients
            for client in self.clients:
                # client.scheduler.update(i, 0.0)
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()
            if i % 100 == 0:
                self.save_global_model_middle(i)

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss), personalized_acc)

        self.save_results()
        self.save_global_model()
