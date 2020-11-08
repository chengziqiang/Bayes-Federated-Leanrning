import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
# import syft as sy
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random
from functools import partial, reduce
from torch.utils.tensorboard import SummaryWriter
import pytz
from datetime import datetime
from scipy.stats import pearsonr
from pkgs import *

UTC_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%b_%d_%H:%M')
file_name = __file__.split('/')[-1].split('.')[1]
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

def train(args):
    seed, workers, optim, funcName, sample_num, noise_tol, prior_var, a, b = args
    set_seed(seed)
    # suffix = "{}_{}_{}_{}_{}_{}_{}_{}:{}".format(file_name, ''.join(workers), optim, funcName, sample_num, noise_tol, prior_var, a, b)
    suffix = "{}_{}sample{}noisy{}prior{}_{}:{}".format(optim, funcName, sample_num, noise_tol, prior_var, a, b)
    path = path_init(os.getcwd()+root_path.format(seed), suffix)
    logger = Logger(log_columns ,path["log"], header=str(args))
    writer = SummaryWriter(path["tensorboard"])
    model = BayesNet(funcName = funcName, prior_var = prior_var, a = a, b = b,).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=.01)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=.001, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=0.00001)
    best = {'val_loss': 9e8, 'epoch': 0}

    train = []
    val_loader = {}
    val_weighted = []
    val_data = {}
    val_label = {}
    for i in range(len(workers)):
        train.append(train_data[workers[i]].sample(frac=0.9, random_state=seed))
        df = train_data[workers[i]].drop(train[i].index)
        val_weighted.append(len(df))
        val_data[workers[i]] = torch.Tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
        label = torch.Tensor(np.reshape(df['y'].values, (-1, 1))).float().to(device).view(-1, 1)
        val_label[workers[i]] = df['y'].values.reshape(-1)
        val_loader[workers[i]] = DataLoader(TensorDataset(val_data[workers[i]], label), batch_size=5000)
    df = pd.concat(train)
    data = torch.Tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
    label = torch.Tensor(df['y'].values).float().to(device).view(-1, 1)
    train_loader = DataLoader(TensorDataset(data, label), batch_size=batch_size, shuffle=True, drop_last=True)
    val_weighted = np.array(val_weighted) / sum(val_weighted)
    outputs = torch.zeros(len(train_loader), batch_size)
    log_likes = torch.zeros(len(train_loader), batch_size)

    for epoch in range(500):
        losses = []
        model.sample(False)
        model.train()
        grad_weight = torch.Tensor([1 / sample_num]).to(device)
        for batch_num, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            log_prior, log_post, log_like, _ = model.sample_elbo(data, label, sample_num)
            loss_sum = log_post - log_prior - log_like
            loss_sum.backward()
            losses.append(list(map(lambda x: x.detach().cpu(), [log_prior, log_post, log_like])))
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()# 可能写在循环外 也就是不设batch 结果还挺好 如果放进去效果不好可研究
        losses = np.array(losses).mean(axis=0)
        log_buffers = [epoch, losses.mean(), losses[0], losses[1], losses[2]]

        loss_sum = []
        correlation = []
        uncertains = []
        model.eval()
        model.sample()
        for worker in workers:
            for batch_num, (data, label) in enumerate(val_loader[worker]):
                # pred = model(data)
                _, _, _, outputs = model.sample_elbo(data, label, 10)
                lower_bound = np.percentile(outputs.cpu().numpy(), 25, axis=0)
                upper_bound = np.percentile(outputs.cpu().numpy(), 75, axis=0)
                uncertain = (upper_bound - lower_bound)
                loss = np.abs(val_label[worker] - (lower_bound + upper_bound) / 2)
                correlation.append(pearsonr(uncertain, loss)[0])
                uncertains.append(uncertain.mean())
                loss_sum.append(loss.mean())
        loss_sum = np.array(loss_sum)
        scheduler.step((loss_sum * val_weighted).sum())
        loss_sum = np.array(loss_sum)
        log_buffers.extend([(loss_sum * val_weighted).sum(), np.mean(uncertains), np.mean(correlation)])
        log_buffers.extend(loss_sum)
        logger.pf(*log_buffers)
        for i in range(1, len(log_columns)):
            writer.add_scalars(log_columns[i],{suffix:log_buffers[i]}, global_step=epoch)

        # if epoch % 30 == 0:
        # model_weights_hist(model, path["weight"]+'/'+filename_prefix, epoch)

        if best["val_loss"] > (loss_sum * val_weighted).sum():
            best["val_loss"] = (loss_sum * val_weighted).sum()
            best["epoch"] = epoch
            torch.save(model.state_dict(), path["model"])
        # if best["epoch"] < epoch - 30:
        #     break
    print(best)
    writer.close()
    with open(path["log"], 'a')as f:
        f.write(str(best))
        f.write('\n'+'#'*10+"train end"+'#'*10+'\n')
        f.write(test(path["model"]))
        f.write('#'*10+"test end"+'#'*10)

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
        loss = (labels - (lower_bound + upper_bound) / 2)**2
        correlation = pearsonr(uncertain, loss)[0]
        uncertain = uncertain.mean()
        loss = loss.mean()
        buffers += "{}: losses : {:.4f}   uncertain: {:.4f}  correlation: {:.4f}\n".format(worker, loss, uncertain, correlation)
    return buffers


if __name__ == '__main__':
    import multiprocessing
    # from multiprocessing import Pool
    # multiprocessing.set_start_method('spawn')
    # pool = Pool(1)
    # map_list = []
    # for arg1 in seed_list:
    #     for arg2 in client_selection(client_list):
    #         for arg3 in init_list:
    #             # for arg4 in np.linspace(0.1,0.9,9).astype(dtype=np.float16):
    #             for arg4 in [0.9]:
    #                 map_list.append((arg1, arg2, arg3, arg4))
    # pool.map(train, map_list)

    # param: seed, workers, funcName, sample_num, noise_tol, prior_var, a, b
    train((0,client_list, "Adam", "RReLU", 1,     .01,         0.01,     0., 1e-5))