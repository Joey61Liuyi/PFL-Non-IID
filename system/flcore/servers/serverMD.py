import wandb
from flcore.clients.clientMD import clientFedMD
from flcore.servers.serverbase import Server
from torch.utils.data import DataLoader
import copy
from utils.data_utils import read_client_data
from threading import Thread
import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np

class FedMD(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                 num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold, run_name, choose_client, step_alignment):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                         num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal,
                         time_threthold, run_name)
        # select slow clients
        self.set_slow_clients()
        for i, train_slow, send_slow in zip(choose_client, self.train_slow_clients, self.send_slow_clients):
            # train, test = read_client_data(dataset, i)
            client = clientFedMD(device, i, train_slow, send_slow, self.train_all[i], self.test_all[i], model[i], batch_size, learning_rate, local_steps)
            self.clients.append(client)
        self.device = device
        self.public_data_loader = DataLoader(self.public, batch_size, drop_last=True)
        self.alignment_step = step_alignment
        del(self.train_all)
        del(self.test_all)
        del(self.public)
        self.iter_trainloader = iter(self.public_data_loader)


        print(f"\nJoin clients / total clients: {self.join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

    def nas_competetive_output(self, output):

        if isinstance(output, tuple):
            output = output[1]

        if isinstance(output, list):
            assert len(output) == 2, "output must has {:} items instead of {:}".format(
                2, len(output)
            )
            output, output_aux = output
        else:
            output, output_aux = output, None

        return output

    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (x, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.public_data_loader)
            (x, y) = next(self.iter_trainloader)

        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)

        return x, y

    def train(self):
        for i in range(self.start_epoch, self.global_rounds+1):
            print(f"\n-------------Round number: {i}-------------")
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

            if i >= self.global_rounds/5:
                for step in range(self.alignment_step):
                    x, y = self.get_next_train_batch()
                    logits = []
                    for client in self.clients:
                        logits.append(copy.deepcopy(client.predict(x)))
                    aggregated_logits = []
                    for j in range(len(logits[0])):
                        tep = None
                        for k in range(len(logits)):
                            if tep == None:
                                tep = logits[k][j]
                            else:
                                tep += logits[k][j]
                        tep /=len(self.clients)
                        aggregated_logits.append(tep)

                    for client in self.clients:
                        client.MD_aggregation(x, aggregated_logits)


            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]


            if i % 100 == 0:
                self.save_global_model_middle(i)

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(self.rs_train_acc), min(self.rs_train_loss), max(self.rs_personalized_acc))
        self.save_results()
        self.save_global_model()