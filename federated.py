import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import OrderedDict
from torch.utils.data import TensorDataset, DataLoader
import random
import os
import time
from functools import partial
from torch.utils.tensorboard import SummaryWriter
import pytz
from datetime import datetime
from scipy.stats import pearsonr
from pkgs import *

UTC_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%b_%d_%H:%M')
file_name = __file__.split('/')[-1].split('.')[0]
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_columns =  ["client"] + log_columns
root_path = f"/result/{version}/"

train_data, test_data, test_loader = {}, {}, {}
for worker in client_list:
    train_data[worker] = pd.read_csv("data/dataset_{}.csv".format(worker))
for worker in client_list:
    df = train_data[worker].sample(frac=0.1, random_state=0)
    train_data[worker] = train_data[worker].drop(df.index)
    data = torch.Tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
    label = torch.Tensor(df['y'].values).float().to(device).view(-1, 1)
    test_loader[worker] = DataLoader(TensorDataset(data, label), batch_size=5000, shuffle=True)
# for worker in ['a','b','c']:
#     df_d = df_d[df_d.SMILES.isin(globals()["df_{}".format(worker)].SMILES.values)]     #remain same part as abc

def train(args):
    seed, workers, aggregate_num, optim, funcName, sample_num, noise_tol, prior_var, a, b = args
    set_seed(seed)
    models, optimizers, train_loader, val_loader, schedulers, logger, writer, path, best = {}, {}, {}, {}, {}, {}, {}, {}, {}
    suffix = "{}_{}_every{}:{{}}".format(file_name, ''.join(workers), aggregate_num)
    val_weighted =[]
    val_label = {}
    for worker in workers:
        best[worker] = {"val_loss": 9e8, "epoch": 0}
        path[worker] = path_init(os.getcwd()+root_path.format(seed), suffix.format(worker))
        logger[worker] = Logger(log_columns ,path[worker]["log"], header=str(args))
        writer[worker] = SummaryWriter(path[worker]["tensorboard"])

        df = train_data[worker].sample(frac=0.9, random_state=seed)
        data = torch.Tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
        label = torch.Tensor(df['y'].values).float().to(device).view(-1, 1)
        train_loader[worker] = DataLoader(TensorDataset(data, label), batch_size=batch_size, shuffle=True)

        df = train_data[worker].drop(df.index)
        data = torch.Tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
        label = torch.Tensor(df['y'].values).float().to(device).view(-1, 1)
        val_label[worker] = df['y'].values.reshape(-1)

        val_loader[worker] = DataLoader(TensorDataset(data, label), batch_size=5000, shuffle=False)
        val_weighted.append(len(df))

        models[worker] = BayesNet().to(device)
        # optimizers[worker] = torch.optim.SGD(params=models[worker].parameters(), lr=0.05,
        #                                         # momentum=0.7
        #                                                     )
        optimizers[worker] = torch.optim.Adam(params=models[worker].parameters(), lr=.05)
        schedulers[worker] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[worker], 'min', factor=0.5, patience=6, min_lr=0.00001)

        # degree.append(torch.tensor(max(1, round(train_data[worker].size()[0] / batch_size))))
    val_weighted = np.array(val_weighted) / sum(val_weighted)

    for epoch in range(1000):
        log_buffers = {}
        for worker in workers:
            losses = []
            optimizer = optimizers[worker]
            model = models[worker]
            model.train()
            for batch_num, (data, label) in enumerate(train_loader[worker]):
                optimizer.zero_grad()
                log_prior, log_post, log_like, _ = model.sample_elbo(data, label, sample_num)
                loss_sum = log_post - log_prior - log_like
                loss_sum.backward()
                losses.append(list(map(lambda x: x.detach().cpu(), [log_prior, log_post, log_like])))
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()# 可能写在循环外 也就是不设batch 结果还挺好 如果放进去效果不好可研究
            losses = np.array(losses).mean(axis=0)
            log_buffers[worker] = [worker, epoch, losses.mean(), losses[0], losses[1], losses[2]]
        if epoch % aggregate_num == 0 and epoch != 0:
            with torch.no_grad():
                aggragate_model(models,True)

        loss_sum = []
        correlation = {}
        uncertains = {}

        for worker in workers:
            outputs = []
            model = models[worker]
            model.eval()
            model.sample()
            for batch_num, (data, label) in enumerate(val_loader[worker]):
                _, _, _, output = model.sample_elbo(data, label, 60)
                outputs.append(output.cpu().numpy())
            outputs = np.concatenate(outputs, axis=1)
            lower_bound = np.percentile(outputs, 25, axis=0)
            upper_bound = np.percentile(outputs, 75, axis=0)
            uncertain = (upper_bound - lower_bound)
            loss = (val_label[worker] - (lower_bound + upper_bound) / 2) **2
            correlation[worker] = pearsonr(uncertain, loss)[0]
            uncertains[worker] = uncertain.mean()
            loss_sum.append(loss.mean())
            schedulers[worker].step(np.array(loss_sum).mean())

        for worker in workers:
            log_buffers[worker].append((loss_sum * val_weighted).sum())
            log_buffers[worker].append(uncertains[worker])
            log_buffers[worker].append(correlation[worker])
            log_buffers[worker].extend(loss_sum)
        for worker in workers:
            logger[worker].pf(*log_buffers[worker])
            for i in range(2, len(log_columns)):
                writer[worker].add_scalars(log_columns[i],{suffix.format(worker):log_buffers[worker][i]}, global_step=epoch)

        for i in range(len(workers)):
            if best[workers[i]]["val_loss"] > loss_sum[i]:
                best[workers[i]]["val_loss"] = loss_sum[i]
                best[workers[i]]["epoch"] = epoch
                torch.save(models[workers[i]].state_dict(), path[workers[i]]["model"])
            # if epoch % 30 == 0:
            #     model_weights_hist(models[worker], path["weight"]+f"/{filename_prefix}:{worker}",epoch)
        # if epoch - min([best[worker]["epoch"] for worker in workers]) > 30:
        #     break
    print(best)
    for worker in workers:
        writer[worker].close()
        with open(path[worker]["log"], 'a') as f:
            f.write(str(best))
            f.write('\n'+'#'*30+"train end"+'#'*30+'\n')
            f.write(test(path[worker]["model"]))
            f.write('#'*30+"test end"+'#'*30)

def test(model_path):
    buffers = ""
    for worker in client_list:
        model = BayesNet()
        # model.load_state_dict(model_path)
        model.to(device)
        model.eval()
        model.sample()
        outputs, labels = [], []
        for data, label in test_loader[worker]:
            _, _, _, output = model.sample_elbo(data, label, 200)
            outputs.append(output.cpu().numpy())
            labels.append(label.cpu().numpy())
        outputs = np.concatenate(outputs, axis=1)
        labels = np.concatenate(labels, axis=1).reshape(-1)
        lower_bound = np.percentile(outputs, 25, axis=0)
        upper_bound = np.percentile(outputs, 75, axis=0)
        uncertain = (upper_bound - lower_bound)
        loss = (labels - (lower_bound + upper_bound) / 2) **2
        correlation = pearsonr(uncertain, loss)[0]
        uncertain = uncertain.mean()
        loss = loss.mean()
        buffers += "{}: losses : {:.4f}   uncertain: {:.4f}  correlation: {:.4f}\n".format(worker, loss, uncertain, correlation)
    return buffers
if __name__ == '__main__':
    # import multiprocessing
    # from multiprocessing import Pool
    # multiprocessing.set_start_method('spawn')
    # pool = Pool(3)
    # map_list = []
    # for arg1 in seed_list:
    #     for arg2 in client_selection_fed(client_list):
    #         for arg3 in aggregate_epochs:
    #             map_list.append((arg1,arg2,arg3))
    # pool.map(train,map_list)

    # for arg in [1, 10, 50, 100, 200]:
    arg1 = client_list
    arg2 = 1
    train((1,arg1, arg2, "Adam", "Sigmoid", 1,     .1,   0.01,     0., 1e-5))
