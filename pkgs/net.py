import math
from functools import reduce
from typing import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter


class ShareSample():
    _sample = True
    def sample(self, is_sample=True):
        ShareSample._sample = is_sample


class Gaussion():
    def __init__(self, mean, standard_deviation ):
        self.loc = mean
        self.scale = standard_deviation

    def sample(self):
        from torch.distributions import Normal
        distribution = Normal(torch.zeros_like(self.loc), torch.ones_like(self.scale))
        return self.loc + self.sigma.mul(distribution.sample())

    @property
    def mean(self):
        return self.loc

    @property
    def sigma(self):
        return torch.log(torch.exp(self.scale)+1)

    def log_prob(self, value):
        return -math.log(math.sqrt(2*math.pi)) - (torch.log(self.sigma) - ((value-self.loc)**2)/(2*self.sigma**2)).sum()

class BayesLinear(nn.Module, ShareSample):
    def __init__(self, in_features, out_features, prior_var = 0.01, a = .0, b = 1e-5, type = "Normal"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        t1 = torch.Tensor(out_features, in_features).normal_(mean=a, std=b)
        t2 = torch.Tensor(out_features, in_features).normal_(mean=0, std=b)
        t3 = torch.Tensor(out_features).normal_(mean=a, std=b)
        t4 = torch.Tensor(out_features).normal_(mean=0, std=b)
        self.weight_loc = Parameter(t1)
        self.weight_scale = Parameter(t2)
        self.bias_loc = Parameter(t3)
        self.bias_scale = Parameter(t4)
        self.weight_prior_loc = Parameter(t1, requires_grad= False)
        self.weight_prior_scale = Parameter(t2, requires_grad= False)
        self.bias_prior_loc = Parameter(t3, requires_grad= False)
        self.bias_prior_scale = Parameter(t4, requires_grad= False)
        self.normal_loc = Parameter(torch.Tensor([0.]), requires_grad= False)
        self.normal_scale = Parameter(torch.Tensor([1.]), requires_grad= False)
        self.stdNormal = Normal(self.normal_loc,self.normal_scale)
        self.log_prior = 0
        self.log_posterior = 0

    def forward(self, x):
        if self.training or self._sample:
            # sample weight
            w_epsilon = self.stdNormal.sample(self.weight_loc.shape).reshape(self.weight_loc.shape)
            weight = self.weight_loc + torch.log(1+torch.exp(self.weight_scale)) * w_epsilon
            # sample bias
            b_epsilon = self.stdNormal.sample(self.bias_loc.shape).reshape(self.bias_loc.shape)
            bias = self.bias_loc + torch.log(1+torch.exp(self.bias_scale)) * b_epsilon
            self.weight_prior = Normal(self.weight_prior_loc, torch.log(1+torch.exp(self.weight_prior_scale)))
            self.bias_prior = Normal(self.bias_prior_loc, torch.log(1+torch.exp(self.bias_prior_scale)))
            self.log_prior = self.weight_prior.log_prob(weight).sum() + self.bias_prior.log_prob(bias).sum()
            self.weight_post = Normal(self.weight_loc, torch.log(1 + torch.exp(self.weight_scale)))
            self.bias_post = Normal(self.bias_loc, torch.log(1 + torch.exp(self.bias_scale)))
            self.log_post = self.weight_post.log_prob(weight).sum() + self.bias_post.log_prob(bias).sum()
        else:
            weight = self.weight_loc
            bias = self.bias_loc

        return F.linear(x, weight, bias)

    # def set_prior(self, buffers = None):
    #     pass
    #     if buffers is None:
    #         buffers = {}
    #         buffers["weight_prior_loc"] = Parameter(torch.zeros_like(self.weight_loc), requires_grad= False)
    #         buffers["weight_prior_scale"] = Parameter(torch.zeros_like(self.weight_scale), requires_grad= False)
    #         buffers["bias_prior_loc"] = Parameter(torch.zeros_like(self.bias_loc), requires_grad= False)
    #         buffers["bias_prior_scale"] = Parameter(torch.zeros_like(self.bias_scale), requires_grad= False)
    #     self.myRegisterBuffers(buffers)
    #     self.weight_prior = Gaussion(self.weight_prior_loc, self.weight_prior_scale)
    #     self.bias_prior = Gaussion(self.bias_prior_loc, self.bias_prior_scale)

    # def myRegisterBuffers(self, buffers: dict ):
    #     pass
    #     for key in buffers.keys():
    #         self.register_parameter(key, buffers[key])   #TODO: buffer在cuda会重赋值，使得高斯模型中loc scale 无效(初步原因parammeter是包括tensor的类,
    #                                                      # cuda会将tensor重赋值, parameter指向cuda后的tensor, 而buffer中只是一个tensor)
    # 考虑重写成nn.module方法 或 传入self


class BayesNet(nn.Module, ShareSample):
    def __init__(self, num=2048, output_num=1, funcName = "RReLU", noise_tol = .1, prior_var = 0.01, a = .0, b = 1e-10):
        super().__init__()
        self.noise_tol = noise_tol
        self.fc1 = BayesLinear(num, int(num/4), prior_var = prior_var, a = a, b = b,)
        self.fc2 = BayesLinear(int(num/4), int(num/16), prior_var = prior_var, a = a, b = b,)
        self.fc3 = BayesLinear(int(num/16), int(num/128), prior_var = prior_var, a = a, b = b,)
        self.fc4 = BayesLinear(int(num/128), output_num, prior_var = prior_var, a = a, b = b,)
        # self.fc4 = BayesLinear(int(num/8), int(num/32), var = prior_var, a = a, b = b,)
        # self.fc6 = BayesLinear(6, output_num, var = prior_var, a = a, b = b,)
        func = getattr(nn, funcName)
        self.r1 = func()
        self.r2 = func()
        self.r3 = func()
        self.r4 = func()
        self.r5 = func()
        self.bn1 = nn.BatchNorm1d(int(num*2))
        self.bn2 = nn.BatchNorm1d(int(num/2))
        self.model = [self.fc1, self.fc2, self.fc3, self.fc4]
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
        # x = self.r3(x)
        # x = self.fc5(x)
        # x = self.r4(x)
        # x = self.fc6(x)
        return x

    def sample_elbo(self, input, target, samples, ):
        # we calculate the negative elbo, which will be our loss function
        #initialize tensors
        device = target.device
        outputs = torch.zeros(samples, target.shape[0]).to(device)
        log_priors = torch.zeros(samples).to(device)
        log_posts = torch.zeros(samples).to(device)
        log_likes = torch.zeros(samples).to(device)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1) # make predictions
            log_priors[i] = self.log_prior() # get log prior
            log_posts[i] = self.log_post() # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).mean() # calculate the log likelihood
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        return log_prior, log_post, log_like, outputs.detach()

    def log_prior(self):
        return reduce(lambda x, y: x+y, [i.log_prior for i in self.model]) / len(self.model)

    def log_post(self):
        return reduce(lambda x, y: x+y, [i.log_post for i in self.model]) / len(self.model)


class ToyNetC(nn.Module, ShareSample):
    def __init__(self, funcName="RReLU", noise_tol=.1, prior_var=0.01, a=.0, b=1e-10):
        super().__init__()
        self.noise_tol = noise_tol
        self.fc1 = BayesLinear(28*28, 512, prior_var = prior_var, a = a, b = b,)
        self.fc2 = BayesLinear(512, 128, prior_var = prior_var, a = a, b = b,)
        self.fc3 = BayesLinear(128, 10, prior_var = prior_var, a = a, b = b,)
        func = getattr(nn, funcName)
        self.r1 = func()
        self.r2 = func()
        self.model = [self.fc1, self.fc2, self.fc3]

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
        device = target.device
        outputs = torch.zeros(samples, target.shape[0], 10).to(device)
        log_priors = torch.zeros(samples).to(device)
        log_posts = torch.zeros(samples).to(device)
        log_likes = torch.zeros(samples).to(device)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            # log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).mean()  
        log_likes = F.nll_loss(outputs.mean(0), target, reduction="none")
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        return log_prior, log_post, log_like, outputs.detach()

    def log_prior(self):
        return reduce(lambda x, y: x+y, [i.log_prior for i in self.model]) / len(self.model)

    def log_post(self):
        return reduce(lambda x, y: x+y, [i.log_post for i in self.model]) / len(self.model)

class ToyNetL(nn.Module, ShareSample):
    def __init__(self, funcName="RReLU", noise_tol=.1, prior_var=0.01, a=.0, b=1e-10):
        super().__init__()
        self.noise_tol = noise_tol
        self.fc1 = BayesLinear(1, 50, prior_var = prior_var, a = a, b = b,)
        self.fc2 = BayesLinear(50, 1, prior_var = prior_var, a = a, b = b,)
        # self.fc3 = BayesLinear(10, 1, prior_var = prior_var, a = a, b = b,)
        func = getattr(nn, funcName)
        self.r1 = func()
        self.r2 = func()
        self.model = [self.fc1, self.fc2]

    def forward(self, x):
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        # x = self.r2(x)
        # x = self.fc3(x)
        return x

    def sample_elbo(self, input, target, samples):
        # we calculate the negative elbo, which will be our loss function
        device = target.device
        outputs = torch.zeros(samples, target.shape[0]).to(device)
        log_priors = torch.zeros(samples).to(device)
        log_posts = torch.zeros(samples).to(device)
        log_likes = torch.zeros(samples).to(device)
        # make predictions and calculate prior, posterior, and likelihood for a given number of samples
        for i in range(samples):
            outputs[i] = self(input).reshape(-1)  # make predictions
            log_priors[i] = self.log_prior()  # get log prior
            log_posts[i] = self.log_post()  # get log variational posterior
            log_likes[i] = Normal(outputs[i], self.noise_tol).log_prob(target.reshape(-1)).sum()  
        # calculate monte carlo estimate of prior posterior and likelihood
        log_prior = log_priors.mean()
        log_post = log_posts.mean()
        log_like = log_likes.mean()
        # calculate the negative elbo (which is our loss function)
        return log_prior, log_post, log_like, outputs.detach()

    def log_prior(self):
        return reduce(lambda x, y: x+y, [i.log_prior for i in self.model])

    def log_post(self):
        return reduce(lambda x, y: x+y, [i.log_post for i in self.model])


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
