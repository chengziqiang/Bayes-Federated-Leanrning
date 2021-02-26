# from pkgs import *
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from pkgs import ToyNetC
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self,):
        super().__init__()
        self.fc1 = nn.Linear(28*28,512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        self.bn1 = nn.BatchNorm1d(128)
        self.dt1 = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dt1(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(0)

workers = list(range(0,8,2))

train_loader, test_loader = {}, {}
train_data= datasets.MNIST('./data', train=True, download=False, transform=transforms.Compose([transforms.ToTensor(),]))
test_data = datasets.MNIST('./data', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(),]))
train_label, test_label = train_data.targets, test_data.targets
train_data, test_data = train_data.data.float()/255, test_data.data.float()/255
for worker in workers:
    index = (train_label >= worker) & (train_label < worker+3)
    train_loader[worker] = DataLoader(TensorDataset(train_data[index].to(device), train_label[index].to(device)), batch_size=200, shuffle=True)
    index = (test_label >= worker) & (test_label < worker+3)
    test_loader[worker] = DataLoader(TensorDataset(test_data[index].to(device), test_label[index].to(device)), batch_size=10000, shuffle=True)
train_loader = train_loader[workers[1]]
# test_loader = test_loader[workers[0]]
# train_loader = DataLoader(datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
#     batch_size=1, shuffle=True,)
# test_loader = DataLoader(datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
#     batch_size=1, shuffle=False,)

epochs = 1000
best = -9e8
net = ToyNetC(funcName="ReLU", noise_tol=.1, prior_var=1., a=.0, b=1e-5).to(device)
net1 = torch.load("net.pt")
new_param_dict = OrderedDict()
for i in [1,2,3]:
    for j in ["weight", "bias"]:
        for k in ["loc", "scale"]:
            s1 = "fc{}.{}_prior_{}"
            s2 = "fc{}.{}_{}"
            new_param_dict[s1.format(i, j, k)] = net1.state_dict()[s2.format(i, j, k)]
            new_param_dict[s2.format(i, j, k)] = net1.state_dict()[s2.format(i, j, k)]
param_dict = net.state_dict()
param_dict.update(new_param_dict)
net.load_state_dict(param_dict)
optimizer = Adam(net.parameters(), lr=0.0001)
for epoch in range(epochs):
    with torch.no_grad():
        net.eval()
        accuracy = []
        for worker in workers:
            acc = 0
            for data, label in test_loader[worker]:
                data, label = data.to(device), label.to(device)
                _, _, _, outputs = net.sample_elbo(data, label, 100)
                acc += (torch.argmax(outputs.mean(0), dim=1) == label).sum().float()
            print(worker, " val accuracy: ", acc.item()/len(test_loader[worker].dataset))
            accuracy.append(acc.item()/len(test_loader[worker].dataset))
        # if best < accuracy[0]:
        #     best = accuracy[0]
        #     torch.save(net, "net.pt")

    net.train()
    losses = []
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        log_prior, log_post, log_like, _ = net.sample_elbo(data, label, 1)
        loss_sum = log_post - 10*log_prior + log_like
        loss_sum.backward()
        losses.append(list(map(lambda x: x.detach().cpu().numpy(), [log_prior, log_post, log_like])))
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
    print("train loss: ", np.array(losses).mean(axis=0))

