from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from functools import reduce

class Gaussion():
    def __init__(self, mean, standard_deviation ):
        self.loc = mean
        self.scale = standard_deviation
    
    def sample(self):
        from torch.distributions.normal import Normal
        distribution = Normal(torch.zeros_like(self.loc), torch.ones_like(self.scale))
        return self.loc + self.sigma.mul(distribution.sample())
    
    @property
    def mean(self):
        return self.loc

    @property
    def sigma(self):
        return torch.log(torch.exp(self.scale)+1)

    def log_prob(self, value):
        return -math.log(math.sqrt(2*math.pi)) - (torch.log(self.sigma) - ((value-self.loc)**2)/(2*self.sigma**2)).mean()

class BayesLinear(nn.Module):
    _sample = False
    @property
    def sample(self):
        return BayesLinear._sample
    
    @sample.setter
    def sample(self, is_sample):
        BayesLinear._sample = is_sample

    def __init__(self, in_features, out_features, type = "Normal"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_loc = Parameter(torch.Tensor(out_features, in_features).normal_(mean=0., std=0.25))
        self.weight_scale = Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.bias_loc = Parameter(torch.Tensor(out_features).normal_(mean=0., std=0.25))
        self.bias_scale = Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        #
        self.weight_prior_loc = Parameter(torch.zeros_like(self.weight_loc), requires_grad= False)
        self.weight_prior_scale = Parameter(torch.zeros_like(self.weight_scale), requires_grad= False)
        self.bias_prior_loc = Parameter(torch.zeros_like(self.bias_loc), requires_grad= False)
        self.bias_prior_scale = Parameter(torch.zeros_like(self.bias_scale), requires_grad= False)
        #
        self.weight = Gaussion(self.weight_loc, self.weight_scale)
        self.bias = Gaussion(self.bias_loc, self.bias_scale)       
        self.weight_prior = Gaussion(self.weight_prior_loc, self.weight_prior_scale)
        self.bias_prior = Gaussion(self.bias_prior_loc, self.bias_prior_scale)
        self.log_prior = 0
        self.log_posterior = 0

    def set_prior(self, buffers = None):
        pass
        if buffers is None:
            buffers = {}
            buffers["weight_prior_loc"] = Parameter(torch.zeros_like(self.weight_loc), requires_grad= False)
            buffers["weight_prior_scale"] = Parameter(torch.zeros_like(self.weight_scale), requires_grad= False)
            buffers["bias_prior_loc"] = Parameter(torch.zeros_like(self.bias_loc), requires_grad= False)
            buffers["bias_prior_scale"] = Parameter(torch.zeros_like(self.bias_scale), requires_grad= False)
        self.myRegisterBuffers(buffers)
        self.weight_prior = Gaussion(self.weight_prior_loc, self.weight_prior_scale)
        self.bias_prior = Gaussion(self.bias_prior_loc, self.bias_prior_scale)

    def myRegisterBuffers(self, buffers: dict ):
        pass
        for key in buffers.keys():
            self.register_parameter(key, buffers[key])   #TODO: buffer在cuda会重赋值，使得高斯模型中loc scale 无效(初步原因parammeter是包括tensor的类,
                                                         # cuda会将tensor重赋值, parameter指向cuda后的tensor, 而buffer中只是一个tensor)
                                                         # 考虑重写成nn.module方法 或 传入self

    def forward(self, x):
        if self.training or BayesLinear._sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mean
            bias = self.bias.mean
        if self.training:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_posterior = self.weight.log_prob(weight) + self.bias_prior.log_prob(bias)
        return F.linear(x, weight, bias)

class BayesNet(nn.Module):
    _sample = False
    @property
    def sample(self):
        return BayesNet._sample
    
    @sample.setter
    def sample(self, is_sample):
        BayesNet._sample = is_sample
        BayesLinear.sample = is_sample
        
    def __init__(self, num=2048, output_num=1):
        super().__init__()
        self.fc1 = BayesLinear(num, num*2)
        self.fc2 = BayesLinear(num*2, int(num/2))
        self.fc3 = BayesLinear(int(num/2), int(num/8))
        self.fc4 = BayesLinear(int(num/8), int(num/32))
        self.fc5 = BayesLinear(int(num/32), 6)
        self.fc6 = BayesLinear(6, output_num)
        self.r1 = nn.RReLU()
        self.r2 = nn.RReLU()
        self.r3 = nn.RReLU()
        self.r4 = nn.RReLU()
        self.r5 = nn.RReLU()
        self.bn1 = nn.BatchNorm1d(int(num*2))
        self.bn2 = nn.BatchNorm1d(int(num/2))
        self.model = [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]
        # self.model = [self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]
    def forward(self, x):
        x = self.fc1(x)
        x = self.r1(x)
        # x = self.bn1(x)
        x = self.fc2(x)
        x = self.r2(x)
        # x = self.bn2(x)
        x = self.fc3(x)
        x = self.r2(x)
        x = self.fc4(x)
        x = self.r3(x)
        x = self.fc5(x)
        x = self.r4(x)
        x = self.fc6(x)
        return x

    def log_prior(self):
        return reduce(lambda x, y: x+y, [i.log_prior for i in self.model]) / len(self.model)
        
    def log_posterior(self):
        return reduce(lambda x, y: x+y, [i.log_posterior for i in self.model]) / len(self.model)


class KLLoss(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, input, target):
        pass




class Net(nn.Module):
    
    def __init__(self, embedding=None, num=2048, output_num=1):
        super(Net, self).__init__()
        # self.fc0_1 = nn.Linear(2048,2048)
        # self.bn0_1 = nn.BatchNorm1d(2048)
        # self.fc0_2 = nn.Linear(2048,2048)
        # self.bn0_2 = nn.BatchNorm1d(2048)
        # self.fc0_3 = nn.Linear(2048,2048)
        # self.bn0_3 = nn.BatchNorm1d(2048)
        if embedding is None:
            self.fc1 = nn.Linear(num, num*2)
        else:
            self.fc1 = nn.Linear(num+embedding,num*2)

        # self.bn1 = nn.BatchNorm1d(num*2)
        self.fc2 = nn.Linear(num*2, int(num/2))
        # self.bn2 = nn.BatchNorm1d(int(num/2))
        self.fc3 = nn.Linear(int(num/2), int(num/8))
        # self.bn3 = nn.BatchNorm1d(int(num/8))
        # self.drop1 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(int(num/8), int(num/32))
        # self.bn4 = nn.BatchNorm1d(int(num/32))
        # self.drop2 = nn.Dropout(0.2)
        self.fc5 = nn.Linear(int(num/32), 6)
        self.fc6 = nn.Linear(6, output_num)

    def forward(self, x):
        # identity = x
        # x = self.fc0_1(x)
        # x = F.relu(self.bn0_1(x + identity))
        # identity = x
        # x = self.fc0_2(x)
        # x = F.relu(self.bn0_2(x + identity))
        # identity = x
        # x = self.fc0_3(x)
        # x = F.relu(self.bn0_3(x + identity))

        x = self.fc1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = self.bn3(x)
        x = F.relu(x)
        # x = self.drop1(x)
        x = self.fc4(x)
        # x = self.bn4(x)
        x = F.relu(x)
        # x = self.drop2(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)

        return x