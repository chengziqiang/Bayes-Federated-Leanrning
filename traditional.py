import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
# import syft as sy
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random
from functools import partial, reduce
from torch.utils.tensorboard import SummaryWriter
import pytz
from datetime import datetime
from pkgs import *

UTC_time = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%b_%d_%H:%M')
file_name = __file__.split('/')[-1].split('.')[0]
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
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

def train(args):
    training = True
    seed, workers, init_type, val_radio = args
    set_seed(seed)
    sample_num, lr = 20, 0.1
    print("sample_num: {}|learning rate: {}|net: {}".format(sample_num, lr, "bayes"))
    filename_prefix =  f"only{''.join(workers)}_radio{val_radio}_{init_type}_seed{seed}"
    path = path_init(os.getcwd()+f"/result/{version}/")

    if training:
        log_columns = ["epoch", "train loss", "train likehood", "train prior", "train postprior", "val likehood", "val likehood a",
                        "val likehood b", "val likehood c", "val likehood d"]
        logger = Logger(log_columns ,path["log"]+f'/{filename_prefix}.txt', header=str(args))
        writer = SummaryWriter(path["tensorboard"]+'/'+file_name)
        model = BayesNet().to(device)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=0.7)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=0.001)
        best = {'val_loss': 9e8, 'epoch': 0}

        train = []
        val_loader = {}
        val_weighted = []
        for i in range(len(workers)):
            train.append(train_data[workers[i]].sample(frac=val_radio, random_state=seed))
            df = train_data[workers[i]].drop(train[i].index)
            val_weighted.append(len(df))
            data = torch.tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
            label = torch.tensor(np.reshape(df['y'].values, (-1, 1))).float().to(device).view(-1, 1)
            val_loader[workers[i]] = DataLoader(TensorDataset(data, label), batch_size=5000, shuffle=True)
        df = pd.concat(train)
        data = torch.tensor(df[['x' + str(i) for i in range(2048)]].values).float().to(device)
        label = torch.tensor(df['y'].values).float().to(device).view(-1, 1)
        train_loader = DataLoader(TensorDataset(data, label), batch_size=batch_size, shuffle=True)
        val_weighted = np.array(val_weighted) / sum(val_weighted)
        for epoch in range(2000):
            losses = []
            model.train()
            grad_weight = torch.Tensor([1 / sample_num]).to(device)
            for batch_num, (data, label) in enumerate(train_loader):
                for _ in range(sample_num):
                    pred = model(data)
                    loss_likehood = F.mse_loss(label, pred) 
                    loss_prior = model.log_prior() 
                    loss_posterior = model.log_posterior() 
                    loss_sum = (loss_likehood - loss_prior + loss_posterior) / sample_num
                    loss_sum.backward()
                    losses.append(list(map(lambda x: x.data.cpu().numpy(), [loss_likehood, -loss_prior, loss_posterior])))
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()# 可能写在循环外 也就是不设batch 结果还挺好 如果放进去效果不好可研究
                optimizer.zero_grad()
            losses = np.array(losses).mean(axis=0)
            log_buffers = [epoch, losses.mean(), losses[0], losses[1], losses[2]]
            
            loss_sum = []
            model.eval()
            for worker in workers:
                losses = []
                for batch_num, (data, label) in enumerate(val_loader[worker]):
                    pred = model(data)
                    loss = F.mse_loss(label, pred, reduction='none')
                    losses.extend(loss.detach().cpu().numpy())
                losses = np.array(losses).mean()
                loss_sum.append(losses)
            scheduler.step((loss_sum * val_weighted).sum())
            loss_sum = np.array(loss_sum) 
            log_buffers.append((loss_sum * val_weighted).sum())
            log_buffers.extend(loss_sum)
            logger.pf(*log_buffers)
            for i in range(1, len(log_columns)):
                writer.add_scalars(log_columns[i],{filename_prefix:log_buffers[i]}, global_step=epoch)

            # if epoch % 30 == 0:
                # model_weights_hist(model, path["weight"]+'/'+filename_prefix, epoch)

            if best["val_loss"] > np.array(losses).mean():
                best["val_loss"] = np.array(losses).mean()
                best["epoch"] = epoch
                torch.save(model.state_dict(), path["model"]+f"/{filename_prefix}.pt")
            if best["epoch"] < epoch - 50:
                break
        print(best)
        writer.close()
        with open(path["log"]+f'/{filename_prefix}.txt', 'a')as f:
            f.write(str(best))
            f.write('#'*10+"train end"+'#'*10+'\n')
        
    with open(path["log"]+f'/{filename_prefix}.txt', 'a')as f:
        f.write(test(path["model"]+f"/{filename_prefix}.pt"))
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
        losses = np.array(losses)
        buffers += "{} : {:.4f}+-{:.4f}\n".format(worker, (losses.max() + losses.min()) / 2, ((losses.max() - losses.min()) / 2))
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

    train((0,client_list,'none',0.9))