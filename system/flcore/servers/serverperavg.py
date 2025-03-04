import copy
from flcore.clients.clientperavg import clientPerAvg
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import wandb

class PerAvg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                 num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, time_threthold, 
                 beta, run_name):
        super().__init__(dataset, algorithm, model, batch_size, learning_rate, global_rounds, local_steps, join_clients,
                         num_clients, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, time_select, goal, 
                         time_threthold, run_name)
        # select slow clients
        self.set_slow_clients()

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            # train, test = read_client_data(dataset, i)
            client = clientPerAvg(device, i, train_slow, send_slow, self.train_all[i], self.test_all[i], model, batch_size,
                                  learning_rate, local_steps, beta)
            self.clients.append(client)

        print(f"\nJoin clients / total clients: {self.join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        for i in range(self.start_epoch, self.global_rounds+1):
            # send all parameter for clients
            print(f"\n-------------Round number: {i}-------------")
            self.send_models()
            if i < self.global_rounds / 2:
                eval_gap = 50
            elif i < self.global_rounds * 95 / 100 and i >= self.global_rounds / 2:
                eval_gap = 20
            else:
                eval_gap = 1
            if i % eval_gap == 0:
                print("\nEvaluate global model")
                test_acc, train_acc, train_loss, personalized_acc = self.evaluate_one_step(i)
                info_dict = {
                    "learning_rate": self.clients[0].optimizer.state_dict()['param_groups'][0]['lr'],
                    "global_valid_top1_acc": test_acc * 100,
                    "average_valid_top1_acc": personalized_acc * 100,
                    "epoch": i
                }
                # print(info_dict)
                wandb.log(info_dict)

            # choose several clients to send back upated model to server
            self.selected_clients = self.clients
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

        print("\nBest personalized results.")
        self.print_(max(self.rs_test_acc), max(
            self.rs_train_acc), min(self.rs_train_loss))

        self.save_results()
        if i % 100 == 0:
            self.save_global_model_middle(i)


    def evaluate_one_step(self, epoch):
        models_temp = []
        for c in self.clients:
            models_temp.append(copy.deepcopy(c.model))
            c.train_one_step()

        stats = self.test_accuracy(epoch)
        stats_train = self.train_accuracy_and_loss()

        # set local model back to client for training process
        for i, c in enumerate(self.clients):
            c.clone_model(models_temp[i], c.model)

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_acc = sum(stats_train[2])*1.0 / sum(stats_train[1])
        train_loss = sum(stats_train[3])*1.0 / sum(stats_train[1])
        personalized_acc = stats[3]
        
        self.rs_test_acc.append(test_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        self.print_(test_acc, train_acc, train_loss, personalized_acc)
        return test_acc, train_acc, train_loss, personalized_acc
