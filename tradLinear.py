#%%
import seaborn as sns
sns.set_theme(style="darkgrid")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from pkgs import *
from torch.distributions import Normal
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import os

import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
np.random.seed(0)
data_x = np.random.rand(100) / 2
noisy = np.random.normal(100) /100
data_y = data_x + 0.3 * np.sin(2*np.pi*(data_x+noisy)) + 0.3 * np.sin(4*np.pi*(data_x+noisy)) + noisy
data_x = data_x.reshape(-1,1)
data_y = data_y.reshape(-1,1)
# data_y = toy_function(data_x)

data_x = torch.Tensor(data_x).to(device)
data_y = torch.Tensor(data_y).to(device)
trainloader = DataLoader(TensorDataset(data_x, data_y),  batch_size=50, shuffle=True)
val_x = np.random.rand(100) / 2
noisy = np.random.normal(100) /100
val_y = val_x + 0.3 * np.sin(2*np.pi*(val_x+noisy)) + 0.3 * np.sin(4*np.pi*(val_x+noisy)) + noisy
val_x = val_x.reshape(-1,1)
val_y = val_y.reshape(-1,1)
# val_y = toy_function(val_x)
val_x = torch.Tensor(val_x).to(device)
val_y = torch.Tensor(val_y).to(device)
valloader = DataLoader(TensorDataset(val_x, val_y),  batch_size=50, shuffle=True)
# optimizer = torch.optim.SGD(params=toy.parameters(), lr=0.002, momentum=0.4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, min_lr=0.001)
sample_nums = 30
epoch = 50
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

# net = MLP_BBB(50, prior_var=20).to(device)
net = ToyNetL(funcName="Sigmoid", noise_tol=1., prior_var=10., a=.0, b=2).to(device)
net.train()
optimizer = torch.optim.Adam(net.parameters(), lr=.1)
# optimizer = torch.optim.SGD(net.parameters(), lr=.1)

epochs = 5000
for epoch in range(epochs):  # loop over the dataset multiple times
    optimizer.zero_grad()
    # forward + backward + optimize
    log_prior, log_post, log_like, outputs = net.sample_elbo(input=data_x, target=data_y, samples=1)
    loss = log_post - log_prior - 100*log_like
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 5)
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch: {}/{}'.format(epoch+1,epochs))
        print('Loss:', loss.item())
print('Finished Training')

#%%
samples = 100
# net.sample()
net.eval()
x_tmp = torch.linspace(0,.5,100).reshape(-1,1)
y_samp = np.zeros((samples,100))
for s in range(samples):
    y_tmp = net(x_tmp).detach().numpy()
    y_samp[s] = y_tmp.reshape(-1)
plt.plot(x_tmp.numpy(), np.mean(y_samp, axis = 0), label='Mean Posterior Predictive')
plt.fill_between(x_tmp.numpy().reshape(-1), np.percentile(y_samp, 2.5, axis = 0), np.percentile(y_samp, 97.5, axis = 0), alpha = 0.25, label='95% Confidence')
plt.legend()
plt.scatter(data_x,data_y)
plt.title('Posterior Predictive')
plt.show()

#%%

