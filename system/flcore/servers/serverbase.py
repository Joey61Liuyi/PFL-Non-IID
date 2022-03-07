import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import wandb
import torchvision.transforms as transforms
import torchvision


class Server(object):
    def __init__(self, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                 num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                 time_threthold, run_name):
        # Set up the main attributes
        self.dataset = dataset
        self.global_rounds = global_rounds
        self.local_steps = local_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_model = copy.deepcopy(model)
        self.join_clients = join_clients
        self.num_clients = num_clients
        self.algorithm = algorithm
        self.time_select = time_select
        self.goal = goal
        self.time_threthold = time_threthold
        self.start_epoch = 0

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_models = []

        self.rs_train_acc = []
        self.rs_train_loss = []
        self.rs_test_acc = []
        self.rs_personalized_acc = []

        self.times = times
        self.eval_gap = eval_gap
        self.client_drop_rate = client_drop_rate
        self.train_slow_rate = train_slow_rate
        self.send_slow_rate = send_slow_rate
        self.run_name = run_name

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
        elif dataset == "Cifar100":
            mean = [x / 255 for x in [129.3, 124.1, 112.4]]
            std = [x / 255 for x in [68.2, 65.4, 70.4]]
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
            train_set = torchvision.datasets.CIFAR100(
                "../dataset/{}/rawdata".format(dataset), train=True, transform=train_transform, download=True
            )
            test_set = torchvision.datasets.CIFAR100(
                "../dataset/{}/rawdata".format(dataset), train=False, transform=test_transform, download=True
            )
            assert len(train_set) == 50000 and len(test_set) == 10000
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set.data), shuffle=False)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set.data), shuffle=False)
            for _, train_data in enumerate(trainloader, 0):
                train_set.data, train_set.targets = train_data
            for _, train_data in enumerate(testloader, 0):
                test_set.data, test_set.targets = train_data

        user_data = np.load('./Dirichlet_0.5_Use_valid_False_{}_non_iid_setting.npy'.format(dataset.lower()),
                            allow_pickle=True).item()

        train_all = []
        test_all = []
        for i in range(self.num_clients):
            train_index = user_data[i]["train"] + user_data[i]["test"]
            test_index = user_data[i]["valid"]
            train = []
            test = []
            for index in train_index:
                train.append((train_set.data[index], train_set.targets[index]))
            for index in test_index:
                test.append((test_set.data[index], test_set.targets[index]))
            train_all.append(train)
            test_all.append(test)
        self.train_all = train_all
        self.test_all = test_all


    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        selected_clients = []
        if self.time_select:
            clients_info = []
            for i, client in enumerate(self.clients):
                clients_info.append(
                    (i, client.train_time_cost['total_cost'] + client.send_time_cost['total_cost']))
            clients_info = sorted(clients_info, key=lambda x: x[1])
            left_idx = np.random.randint(
                0, self.num_clients - self.join_clients)
            selected_clients = [self.clients[clients_info[i][0]]
                                for i in range(left_idx, left_idx + self.join_clients)]
        else:
            selected_clients = list(np.random.choice(self.clients, self.join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            if client.send_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))

            client.set_parameters(copy.deepcopy(self.global_model))

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.join_clients))

        active_train_samples = 0
        for client in active_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_models = []
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                self.uploaded_weights.append(client.train_samples / active_train_samples)
                self.uploaded_models.append(copy.deepcopy(client.model))

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() / self.join_clients

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        

    # def aggregate_parameters(self):

    #     for param in self.global_model.parameters():
    #         param.data = torch.zeros_like(param.data)

    #     active_train_samples = 0
    #     for client in self.selected_clients:
    #         active_train_samples += client.train_samples

    #     for client in self.selected_clients:
    #         self.add_parameters(client, client.train_samples / active_train_samples)


    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.global_model, os.path.join(model_path, self.algorithm + "_server" + ".pt"))

    def save_global_model_middle(self, epoch):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        checkpoint_dict = {
            "epoch": epoch,
            "model_dict": self.global_model.state_dict()
        }
        torch.save(checkpoint_dict, os.path.join(model_path, self.run_name + ".pth"))

    def load_model(self, path):
        # model_path = os.path.join("models", self.dataset, "server" + ".pt")
        # assert (os.path.exists(model_path))
        print(os.path)
        info = torch.load(path)
        self.global_model.load_state_dict(info['model_dict'])
        self.start_epoch = info['epoch'] + 1
        # self.global_model = torch.load(model_path)

    # def model_exists(self):
    #     return os.path.exists(os.path.join("models", self.dataset, "server" + ".pt"))

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc) & len(self.rs_train_acc) & len(self.rs_train_loss)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            with h5py.File(result_path + "{}.h5".format(algo), 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    def test_accuracy(self, epoch):
        num_samples = []
        tot_correct = []
        accuracy_list = []
        info_dict = {"epoch": epoch}
        for c in self.clients:
            ct, ns, acc = c.test_accuracy()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            accuracy_list.append(acc)
            info_dict["{}user_a_top1".format(c.id)] = acc

        wandb.log(info_dict)
        mean_acc = np.mean(accuracy_list)
        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, mean_acc

    def train_accuracy_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            ct, cl, ns = c.train_accuracy_and_loss()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, losses

    # evaluate all clients
    def evaluate(self, epoch):
        stats = self.test_accuracy(epoch)
        stats_train = self.train_accuracy_and_loss()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        personalized_acc = stats[3]
        
        self.rs_test_acc.append(test_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        self.rs_personalized_acc.append(personalized_acc)
        self.print_(test_acc, train_acc, train_loss, personalized_acc)
        return test_acc, train_acc, train_loss, personalized_acc


    def print_(self, test_acc, train_acc, train_loss, personalized_acc):
        if personalized_acc:
            print("Average Personalized Test Accurancy: {:.4f}".format(personalized_acc))
        print("Global Test Accurancy: {:.4f}".format(test_acc))
        print("Average Train Accurancy: {:.4f}".format(train_acc))
        print("Average Train Loss: {:.4f}".format(train_loss))
