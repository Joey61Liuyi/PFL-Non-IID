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
                 num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                         num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold)
        # select slow clients
        self.set_slow_clients()



        if dataset == "Cifar10":
            mean = [x / 255 for x in [125.3, 123.0, 113.9]]
            std = [x / 255 for x in [63.0, 62.1, 66.7]]
            lists = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]

            train_transform = transforms.Compose(lists)
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]
            )
            xshape = (1, 3, 32, 32)
            train_set = torchvision.datasets.CIFAR10(
                "../dataset/Cifar10/rawdata", train=True, transform=train_transform, download=True
            )
            test_set = torchvision.datasets.CIFAR10(
                "../dataset/Cifar10/rawdata", train=False, transform=test_transform, download=True
            )

            trainloader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set.data), shuffle=False)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set.data), shuffle=False)
            for _, train_data in enumerate(trainloader, 0):
                train_set.data, train_set.targets = train_data
            for _, train_data in enumerate(testloader, 0):
                test_set.data, test_set.targets = train_data

        user_data = np.load('./Dirichlet_0.1_Use_valid_False_{}_non_iid_setting.npy'.format(dataset),
                            allow_pickle=True).item()

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):

            train_index = user_data[i]["train"] + user_data[i]["test"]
            test_index = user_data[i]["valid"]
            train = []
            test = []
            for index in train_index:
                train.append((train_set.data[index], train_set.targets[index]))
            for index in test_index:
                test.append((test_set.data[index], test_set.targets[index]))

            # train_new, test_new = read_client_data(dataset, i)
            client = clientAVG(device, i, train_slow, send_slow, train, test, model, batch_size, learning_rate, local_steps)
            self.clients.append(client)

        print(f"\nJoin clients / total clients: {self.join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.global_rounds+1):
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                test_acc, train_acc, train_loss = self.evaluate()
                info_dict = {
                    "average_valid_top1_acc": test_acc,
                    "epoch": i
                }
                wandb.log(info_dict)
            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()
            self.save_global_model_middle(i)

        print("\nBest global results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        self.save_global_model()
