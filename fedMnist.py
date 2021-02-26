#%%
# # from pkgs import *
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from pkgs import ToyNetC, aggragate_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(0)
torch.seed = 0
epochs = 1000
aggregate_num = 1

workers = list(range(0,8,2))
models, optimizers, train_loader, test_loader, schedulers, logger, writer, path, best = {}, {}, {}, {}, {}, {}, {}, {}, {}

train_data= datasets.MNIST('./data', train=True, download=False, transform=transforms.Compose([transforms.ToTensor(),]))
test_data = datasets.MNIST('./data', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(),]))
train_label, test_label = train_data.targets, test_data.targets
train_data, test_data = train_data.data.float()/255, test_data.data.float()/255
for worker in workers:
    index = (train_label >= worker) & (train_label < worker+2)
    train_loader[worker] = DataLoader(TensorDataset(train_data[index].to(device), train_label[index].to(device)), batch_size=200, shuffle=True)
    index = (test_label >= worker) & (test_label < worker+2)
    test_loader[worker] = DataLoader(TensorDataset(test_data[index].to(device), test_label[index].to(device)), batch_size=10000, shuffle=True)
    best[worker] = {"val_loss": 9e8, "epoch": 0}
    models[worker] = ToyNetC(funcName="ReLU", noise_tol=.1, prior_var=1., a=.0, b=1e-5).to(device)
    # optimizers[worker] = torch.optim.SGD(params=models[worker].parameters(), lr=0.05, momentum=0.7)
    optimizers[worker] = torch.optim.Adam(params=models[worker].parameters(), lr=.001)
    schedulers[worker] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[worker], 'min', factor=0.5, patience=6, min_lr=0.00001)

#%%
for epoch in range(epochs):
    for worker in workers:
        net, optimizer, data_loader = models[worker], optimizers[worker], train_loader[worker]
        net.train()
        losses = []
        for data, label in data_loader:
            optimizer.zero_grad()
            log_prior, log_post, log_like, _ = net.sample_elbo(data, label, 1)
            loss_sum = log_post - log_prior + log_like
            loss_sum.backward()
            losses.append(list(map(lambda x: x.detach().cpu().numpy(), [log_prior, log_post, log_like])))
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
        print("client: ", worker, "train loss: ", np.array(losses).mean(axis=0))

    with torch.no_grad():
        if epoch % aggregate_num == 0 and epoch != 0:
            aggragate_model(models,True)

        for worker in workers:
            net, scheduler = models[worker], schedulers[worker]
            net.eval()
            result = []
            for key in test_loader.keys():
                losses = []
                acc = 0
                for data, label in test_loader[key]:
                    _, _, _, outputs = net.sample_elbo(data, label, 100)
                    acc += (torch.argmax(outputs.mean(0), dim=1) == label).sum().float()
                result.append((acc * 100 / len(test_loader[key].dataset)).item())
            print("val accuracy: ", worker, result)
