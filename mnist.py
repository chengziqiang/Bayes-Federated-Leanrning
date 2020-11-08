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
from pkgs import ToyNet


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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(0)
train_sets = datasets.MNIST('./data', train=True, download=False, transform=transforms.Compose([transforms.ToTensor(),
            #    transforms.Normalize((0.5,), (1.0,))
        ]))
train_loader = DataLoader(train_sets , batch_size=200, shuffle=True, pin_memory=True)
test_sets = datasets.MNIST('./data', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(),
            #    transforms.Normalize((0.5,), (1.0,))
        ]))
test_loader = DataLoader(test_sets, batch_size=1000, shuffle=True, pin_memory=True)
# test_loader = DataLoader(TensorDataset(test_sets.), batch_size=200, shuffle=True, pin_memory=True)
# train_loader = DataLoader(datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
#     batch_size=1, shuffle=True,)
# test_loader = DataLoader(datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor()),
#     batch_size=1, shuffle=False,)

epochs = 1000
best = 9e8
net = ToyNet(funcName="ReLU", noise_tol=.1, prior_var=1., a=.0, b=1e-5).to(device)
optimizer = Adam(net.parameters(), lr=0.001)
for epoch in range(epochs):
    net.train()
    losses = []
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        log_prior, log_post, log_like, _ = net.sample_elbo(data, label, 1)
        loss_sum = log_post - log_prior + log_like
        loss_sum.backward()
        losses.append(list(map(lambda x: x.detach().cpu().numpy(), [log_prior, log_post, log_like])))
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
    print("train loss: ", np.array(losses).mean(axis=0))

    net.eval()
    losses = []
    acc = 0
    samples_num = 10
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            _, _, _, outputs = net.sample_elbo(data, label, 100)
            acc += (torch.argmax(outputs.mean(0), dim=1) == label).sum().float()
        print("val accuracy: ", acc/len(test_loader.dataset))
