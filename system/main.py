#!/usr/bin/env python
import torch
import argparse
import os
import time
import warnings
import numpy as np
import wandb


from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.servermocha import MOCHA
from flcore.servers.serveramp import FedAMP
from flcore.servers.serverhamp import HeurFedAMP
from flcore.trainmodel.models import *
from flcore.trainmodel.resnet import resnet18 as resnet
from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from models import Networks, obtain_model
from collections import namedtuple
warnings.simplefilter("ignore")
import ast
import re
import random

# hyper-params for Text tasks
vocab_size = 98635
max_len=200
hidden_dim=32


def prepare_seed(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

def run(goal, dataset, num_labels, device, algorithm, model, local_batch_size, local_learning_rate, global_rounds, local_steps, join_clients, 
        num_clients, beta, lamda, K, p_learning_rate, times, eval_gap, client_drop_rate, train_slow_rate, send_slow_rate, 
        time_select, time_threthold, M, mu, itk, alphaK, sigma, xi, genotype):

    time_list = []
    reporter = MemReporter()

    for i in range(times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        Model = None

        # Generate Model
        if model == "mclr":
            if dataset == "mnist" or dataset == "fmnist":
                Model = Mclr_Logistic(1*28*28, num_labels=num_labels).to(device)
            elif dataset == "Cifar10" or dataset == "Cifar100":
                Model = Mclr_Logistic(3*32*32, num_labels=num_labels).to(device)
            else:
                Model = Mclr_Logistic(60, num_labels=num_labels).to(device)

        elif model == "cnn":
            if dataset == "mnist" or dataset == "fmnist":
                Model = LeNet(num_labels=num_labels).to(device)
            elif dataset == "Cifar10" or dataset == "Cifar100":
                Model = CifarNet(num_labels=num_labels).to(device)
            else:
                raise NotImplementedError

        elif model == "dnn": # non-convex
            if dataset == "mnist" or dataset == "fmnist":
                Model = DNN(1*28*28, 100, num_labels=num_labels).to(device)
            elif dataset == "Cifar10" or dataset == "Cifar100":
                Model = DNN(3*32*32, 100, num_labels=num_labels).to(device)
            else:
                Model = DNN(60, 20, num_labels=num_labels).to(device)

        # elif model[:3] == "vgg":
        #     pass
        
        elif model[:6] == "resnet":
            if dataset == "Cifar10" or dataset == "Cifar100":
                # Model = torch.hub.load('pytorch/vision:v0.6.0', model, pretrained=True)
                # Model.fc = ResNetClassifier(input_dim=list(Model.fc.weight.size())[1], num_labels=num_labels)
                # Model.to(device)
                Model = resnet(num_labels=num_labels).to(device)
            else:
                raise NotImplementedError

        elif model == "lstm":
            Model = LSTMNet(hidden_dim=hidden_dim, bidirectional=True, vocab_size=vocab_size, 
                            num_labels=num_labels).to(device)

        elif model == "fastText":
            Model = fastText(hidden_dim=hidden_dim, vocab_size=vocab_size, num_labels=num_labels).to(device)

        elif model == "TextCNN":
            Model = TextCNN(hidden_dim=hidden_dim, max_len=max_len, vocab_size=vocab_size, 
                            num_labels=num_labels).to(device)

        else:
            model_config_dict = {
                "super_type": "infer-nasnet.cifar",
                "genotype": "none",
                "dataset": "cifar",
                "class_num": 10,
                "ichannel": 33,
                "layers": 2,
                "stem_multi": 3,
                "auxiliary": 1,
                "drop_path_prob": 0.2
            }
            Arguments = namedtuple("Configure", " ".join(model_config_dict.keys()))
            content = Arguments(**model_config_dict)
            Model = obtain_model(content, genotype)
            Model = Model.to(device)

        # select algorithm
        if algorithm == "FedAvg":
            server = FedAvg(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, join_clients, num_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold)

        elif algorithm == "PerAvg":
            server = PerAvg(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, join_clients, num_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, beta)

        elif algorithm == "pFedMe":
            server = pFedMe(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, join_clients, num_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, beta, lamda, K, p_learning_rate)

        elif algorithm == "FedProx":
            server = FedProx(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, join_clients, num_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, mu)

        elif algorithm == "FedFomo":
            server = FedFomo(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, join_clients, num_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, M)

        elif algorithm == "MOCHA":
            server = MOCHA(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, join_clients, num_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, itk)

        elif algorithm == "FedAMP":
            server = FedAMP(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, join_clients, num_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, alphaK, lamda, sigma)
        
        elif algorithm == "HeurFedAMP":
            server = HeurFedAMP(device, dataset, algorithm, Model, local_batch_size, local_learning_rate, global_rounds,
                            local_steps, join_clients, num_clients, i, eval_gap, client_drop_rate, train_slow_rate, 
                            send_slow_rate, time_select, goal, time_threthold, alphaK, lamda, sigma, xi)

        del(model)
        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=dataset, algorithm=algorithm, goal=goal, times=times, length=global_rounds/eval_gap+1)

    # Personalization average
    if algorithm == "pFedMe": 
        average_data(dataset=dataset, algorithm=algorithm+'_p', goal=goal, times=times, length=global_rounds/eval_gap+1)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10",
                        choices=["mnist", "synthetic", "Cifar10", "agnews", "fmnist", "Cifar100", \
                        "sogounews"])
    parser.add_argument('-nb', "--num_labels", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="Searched")
    parser.add_argument('-lbs', "--local_batch_size", type=int, default=96)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.025,
                        help="Local learning rate")
    parser.add_argument('-gr', "--global_rounds", type=int, default=120)
    parser.add_argument('-ls', "--local_steps", type=int, default=1)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg",
                        choices=["pFedMe", "PerAvg", "FedAvg", "FedProx", \
                        "FedFomo", "MOCHA", "FedPlayer", "FedAMP", "HeurFedAMP"])
    parser.add_argument('-jc', "--join_clients", type=int, default=5,
                        help="Number of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Dropout rate for clients")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=float("inf"),
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / HeurFedAMP
    parser.add_argument('-bt', "--beta", type=float, default=0.0,
                        help="Average moving parameter for pFedMe, Second learning rate of Per-FedAvg")
    parser.add_argument('-lam', "--lamda", type=float, default=15,
                        help="Regularization weight for pFedMe and FedAMP")
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.015,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # MOCHA
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # HeurFedAMP
    parser.add_argument('-xi', "--xi", type=float, default=1.0)

    config = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

    if config.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        config.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(config.algorithm))
    print("Local batch size: {}".format(config.local_batch_size))
    print("Local steps: {}".format(config.local_steps))
    print("Local learing rate: {}".format(config.local_learning_rate))
    print("Total number of clients: {}".format(config.num_clients))
    print("Clients join in each round: {}".format(config.join_clients))
    print("Client drop rate: {}".format(config.client_drop_rate))
    print("Time select: {}".format(config.time_select))
    print("Time threthold: {}".format(config.time_threthold))
    print("Global rounds: {}".format(config.global_rounds))
    print("Running times: {}".format(config.times))
    print("Dataset: {}".format(config.dataset))
    print("Local model: {}".format(config.model))
    print("Using device: {}".format(config.device))

    if config.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    elif config.algorithm == "pFedMe":
        print("Average moving parameter beta: {}".format(config.beta))
        print("Regularization rate: {}".format(config.lamda))
        print("Number of personalized training steps: {}".format(config.K))
        print("personalized learning rate to caculate theta: {}".format(config.p_learning_rate))
    elif config.algorithm == "PerAvg":
        print("Second learning rate beta: {}".format(config.beta))
    elif config.algorithm == "FedProx":
        print("Proximal rate: {}".format(config.mu))
    elif config.algorithm == "FedFomo":
        print("Server sends {} models to one client at each round".format(config.M))
    elif config.algorithm == "MOCHA":
        print("The iterations for solving quadratic subproblems: {}".format(config.itk))
    elif config.algorithm == "FedAMP":
        print("alphaK: {}".format(config.alphaK))
        print("lamda: {}".format(config.lamda))
        print("sigma: {}".format(config.sigma))
    elif config.algorithm == "HeurFedAMP":
        print("alphaK: {}".format(config.alphaK))
        print("lamda: {}".format(config.lamda))
        print("sigma: {}".format(config.sigma))
        print("xi: {}".format(config.xi))

    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:

    log_dir = "./0.5Dirichlet_Serched_result.log"
    model_owner = 0
    # model_list = ["resnet", "GDAS_V1"]
    genotype_list = {}
    user_list = {}
    user = 0
    for line in open(log_dir):
        if "<<<--->>>" in line:
            tep_dict = ast.literal_eval(re.search('({.+})', line).group(0))
            count = 0
            for j in tep_dict['normal']:
                for k in j:
                    if 'skip_connect' in k[0]:
                        count += 1
            if count == 2:
                # if user%5 not in genotype_list:
                # logger.log("user{}'s architecture is chosen from epoch {}".format(user%5, user//5))
                genotype_list[user % 5] = tep_dict
                user_list[user % 5] = user // 5
            user += 1

    for user in user_list:
        print("user{}'s architecture is chosen from epoch {}".format(user, user_list[user]))
    print(genotype_list)
    model_owner = 0

    if config.model in Networks:
        genotype = Networks[config.model]
        run_name = "{}-{}-{}".format(config.model, config.algorithm, config.dataset)
    else:
        if model_owner!=None:
            genotype = genotype_list[model_owner]
            run_name = "{}-{}-{}-{}".format(config.model, model_owner, config.algorithm, config.dataset)
        else:
            genotype = None

    seed = 666
    prepare_seed(seed)
    wandb_project = "Trial_New"
    # run_name = "{}-{}-{}-{}".format(config.model, config.algorithm, config.dataset, config.local_learning_rate)
    resume_str = None
    wandb.init(project=wandb_project, name=run_name, resume=resume_str)
    run(
        goal=config.goal,
        dataset=config.dataset,
        num_labels=config.num_labels,
        device=config.device,
        algorithm=config.algorithm,
        model=config.model,
        local_batch_size=config.local_batch_size,
        local_learning_rate=config.local_learning_rate,
        global_rounds=config.global_rounds,
        local_steps=config.local_steps,
        join_clients=config.join_clients,
        num_clients=config.num_clients,
        beta=config.beta,
        lamda=config.lamda,
        K=config.K,
        p_learning_rate=config.p_learning_rate,
        times=config.times,
        eval_gap=config.eval_gap,
        client_drop_rate=config.client_drop_rate,
        train_slow_rate=config.train_slow_rate,
        send_slow_rate=config.send_slow_rate,
        time_select=config.time_select,
        time_threthold=config.time_threthold,
        M = config.M,
        mu=config.mu,
        itk=config.itk,
        alphaK=config.alphaK,
        sigma=config.sigma,
        xi=config.xi,
        genotype=genotype
    )
    wandb.finish()


        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
