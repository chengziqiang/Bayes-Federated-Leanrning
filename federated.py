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
from pkgs import *

UTC_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%b_%d_%H:%M')
file_name = __file__.split('/')[-1].split('.')[0]
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data, test_data, test_loader = {}, {}, {}
for worker in client_list:
    train_data[worker] = pd.read_csv("data/dataset_{}.csv".format(worker))
for worker in client_list:
    df = train_data[worker].sample(frac=0.1, random_state=0)
    train_data[worker] = train_data[worker].drop(df.index)
    data = torch.tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
    label = torch.tensor(df['y'].values).float().to(device).view(-1, 1)
    test_loader[worker] = DataLoader(TensorDataset(data, label), batch_size=5000, shuffle=True)
# for worker in ['a','b','c']:
#     df_d = df_d[df_d.SMILES.isin(globals()["df_{}".format(worker)].SMILES.values)]     #remain same part as abc

def train(args):
    seed, need_degree, init_type, workers, aggregate_num, val_radio = args
    filename_prefix = f"{need_degree}_{''.join(workers)}_every{aggregate_num}_radio{val_radio}_{init_type}_seed{seed}"
    path = path_init(os.getcwd()+f"/result/{version}/")
    set_seed(seed)
    models, optimizers, train_loader, val_loader, schedulers, logger, writer = {}, {}, {}, {}, {}, {}, {}
    degree, val_weighted = [], []
    best = {"val_loss": 9e8, "epoch": 0}
    log_columns = ["client", "epoch", "train loss", "train likehood", "train prior", "train postprior", "val likehood", "val likehood a",
                        "val likehood b", "val likehood c", "val likehood d"]
    sample_num = 20

    for worker in workers:
        df = train_data[worker].sample(frac=val_radio, random_state=seed)
        data = torch.tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
        label = torch.tensor(df['y'].values).float().to(device).view(-1, 1)
        train_loader[worker] = DataLoader(TensorDataset(data, label), batch_size=batch_size, shuffle=True)

        df = train_data[worker].drop(df.index)
        data = torch.tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
        label = torch.tensor(df['y'].values).float().to(device).view(-1, 1)
        val_loader[worker] = DataLoader(TensorDataset(data, label), batch_size=5000, shuffle=False)
        val_weighted.append(len(df))

        models[worker] = BayesNet().to(device)
        optimizers[worker] = torch.optim.SGD(params=models[worker].parameters(), lr=0.05, momentum=0.4)
        schedulers[worker] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[worker], 'min', factor=0.5, patience=6, min_lr=0.001)
        logger[worker] = Logger(log_columns ,path["log"]+f'/{filename_prefix}:{worker}.txt', header=str(args))
        writer[worker] = SummaryWriter(path["tensorboard"]+'/'+file_name+'/'+worker)

        # degree.append(torch.tensor(max(1, round(train_data[worker].size()[0] / batch_size))))
    val_weighted = np.array(val_weighted) / sum(val_weighted)

    for epoch in range(2000):
        log_buffers = {}
        for worker in workers:
            losses = []
            optimizer = optimizers[worker]
            model = models[worker]
            model.train()
            for batch_num, (data, label) in enumerate(train_loader[worker]):
                for _ in range(sample_num):
                    pred = model(data)
                    loss_likehood = F.mse_loss(label, pred) 
                    loss_prior = model.log_prior() 
                    loss_posterior = model.log_posterior() 
                    loss_sum = (loss_likehood - loss_prior + loss_posterior) / sample_num
                    loss_sum.backward()
                    losses.append(list(map(lambda x: x.data.cpu().numpy(), [loss_likehood, -loss_prior, loss_posterior])))
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                optimizer.zero_grad()
            losses = np.array(losses).mean(axis=0)
            log_buffers[worker] = [worker, epoch, losses.mean(), losses[0], losses[1], losses[2]]

        last_params = None
        beta = 0.9
        if epoch % aggregate_num == 0 and epoch != 0:
            with torch.no_grad():
                aggragate_model(models)

        loss_sum = []
        for worker in workers:
            losses = []
            for batch_num, (data, label) in enumerate(val_loader[worker]):
                models[worker].eval()
                pred = models[worker](data)
                loss = F.mse_loss(label, pred, reduction='none')
                losses.extend(loss.detach().cpu().numpy())
            losses = np.array(losses).mean()
            loss_sum.append(losses)
            schedulers[worker].step(np.array(losses).mean())
        for worker in workers:
            log_buffers[worker].append((loss_sum * val_weighted).sum())
            log_buffers[worker].extend(loss_sum)

        loss_sum = (loss_sum * val_weighted).sum()  # TODO how to find a best model with loss of each worker without leaky
        
        if epoch % aggregate_num == 0 and epoch != 0:
            if best["val_loss"] > loss_sum:
                best["val_loss"] = loss_sum
                best["epoch"] = epoch
                torch.save(models[workers[0]].state_dict(), path["model"]+f"/{filename_prefix}.pt")
        for worker in workers:
            logger[worker].pf(*log_buffers[worker])
            for i in range(2, len(log_columns)):
                writer[worker].add_scalars(log_columns[i],{filename_prefix:log_buffers[worker][i]}, global_step=epoch)
            # if epoch % 30 == 0:
            #     model_weights_hist(models[worker], path["weight"]+f"/{filename_prefix}:{worker}",epoch)
        if best["epoch"] < epoch - 60:
            break
    print(best)
    for worker in workers:
        writer[worker].close()
        with open(path["log"]+f"/{filename_prefix}-{worker}.txt", 'a') as f:
            f.write(str(best))
            f.write('#'*10+"train end"+'#'*10)
            f.write( (path["model"]+f"/{filename_prefix}.pt"))
            f.write('#'*10+"test end"+'#'*10)



def test(model_path):
    buffers = ""
    for worker in client_list:
        losses = []
        for seed in seed_list:
            model = BayesNet()
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()
            loss = []
            for batch_num, (data, label) in enumerate(test_loader[worker]):
                pred = model(data)
                l = F.mse_loss(label, pred, reduction='none')
                loss.extend(l.detach().cpu().numpy())
            loss = np.array(loss).mean()
            losses.append(loss)
        buffers += "{} : {:.4f}+-{:.4f}\n".format(worker, (losses.max() + losses.min()) / 2, ((losses.max() - losses.min()) / 2))
    return buffers

if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool
    # multiprocessing.set_start_method('spawn')
    # pool = Pool(3)
    # map_list = []
    # for arg1 in seed_list:
    #     # for arg2 in ['degree','no-degree']:
    #     for arg2 in ['degree',]:
    #         for arg3 in init_list:
    #             for arg4 in client_selection_fed(client_list):
    #             # for arg4 in [client_list,]:
    #                 for arg5 in aggregate_epochs:
    #                     for arg6 in [0.9]:
    #                         map_list.append((arg1,arg2,arg3,arg4,arg5,arg6))
    # pool.map(train,map_list)

    for arg in [1, 10, 50, 100, 200]:
        train((1,'degree','none',client_list, arg,0.9))


