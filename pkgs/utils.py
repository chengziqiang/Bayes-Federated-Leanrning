import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings


warnings.filterwarnings("ignore")

def experiment(func):
    def output(arg):
        return [arg]
    return output


def set_seed(init_seed):
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)

def init_weight(layer_name,init_type):
    if type(layer_name) == nn.Linear :

        if init_type == "normal":
            nn.init.normal_(layer_name.weight, mean=0., std=.025)
            nn.init.normal_(layer_name.bias, mean=0., std=.025)
        elif init_type == "uniform":
            nn.init.uniform_(layer_name.weight,a=-0.05, b=.05)
            nn.init.uniform_(layer_name.bias,a=-0.02, b=.02)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(layer_name.weight,gain=1.)
            nn.init.uniform_(layer_name.bias, a=-0.02, b=.02)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(layer_name.weight,gain=1.)
            nn.init.uniform_(layer_name.bias, a=-0.02, b=.02)   #can not processes with fewer than 2 dimensions
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer_name.weight,nonlinearity='relu')
            nn.init.uniform_(layer_name.bias, a=-0.02, b=.02)
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(layer_name.weight,nonlinearity='relu')
            nn.init.uniform_(layer_name.bias, a=-0.02, b=.02)
        elif init_type == "none":
            pass

from functools import reduce
def aggragate_model(models, set_prior=False):
    keys = list(models.keys())
    modelList = [models[key].state_dict() for key in keys]
    index = [i  for i in modelList[0].keys() if find_loc(i)]
    new_state_dict = {}
    for i in index:
        new_state_dict[i] = reduce(lambda x, y: x+y[i], modelList[1:], modelList[0][i]) / len(modelList)
    for i in index:
        i1 = list(i)
        i1.insert(-3, "prior_")
        i1 = "".join(i1)
        for n in range(len(modelList)):
            if set_prior:
                # modelList[n][i1] = new_state_dict[i]
                modelList[n][i1] = modelList[n][i]
            modelList[n][i] = new_state_dict[i]
            #先验为汇聚前（后）模型
            
    for i in range(len(keys)):
        models[keys[i]].load_state_dict(modelList[i])
        

def find_loc(s):
    if s.find("weight_loc") == -1 and s.find("bias_loc") == -1:
        return False
    return True

def model_weights_hist(model,path,epoch):
    sns.set()
    sns.set_style("ticks")
    sns.set_color_codes()
    # ax_num = 0
    # nrows = 2
    # ncols = 3
    # max_num = 6   #需要绘制的权重的层数

    fig, axes = plt.subplots(1, 1, figsize=(4,3))
    for name, parameter in model.named_parameters():
        # if ax_num < max_num and name.find('bn') == -1 and name.find('bias') == -1:
            # ax = sns.distplot(x, color="y")
        
        axes.set_xlim(-1,1)
        parameter = parameter.view(-1, 1).cpu().detach().numpy()
        sns.distplot(parameter,bins=200,kde=False,axlabel=name+f"\n{np.sum(np.abs(parameter)<=0.1)}/{len(parameter)}",ax=axes)
        # sns.distplot(parameter,bins=100,kde=False,ax=ax,color='r')
        # ax_num += 1
        fig.savefig("{}\{}\{}.png".format(path,name,str(epoch).zfill(4)),bbox_inches="tight",pad_inches = 0,dpi=300)

# @experiment
def client_selection_fed(clients):
    yield clients
    # for client in clients:
    #     list_copy = clients.copy()
    #     list_copy.remove(client)
    #     yield list_copy
# @experiment
def client_selection(clients):
    yield clients
    # for client in clients:
    #     yield [client]

def path_init(root_path, suffix):
    root_path = root_path.rstrip('/')
    path = {}
    path["test"] = root_path + '/test' 
    path["train"] = root_path + '/train'
    root_path =  path["train"]
    subPath = ['model', 'log', 'plot', 'tensorboard']
    for sub in subPath:
        path[sub] = root_path + '/' + sub
    for key, value in path.items():
        if not os.path.exists(value):
            os.makedirs(value)
    path["model"] += f"/{suffix}.pt"
    path["log"] += f"/{suffix}.txt"
    return path

def df_precessing(df):
    smiles_list = df.SMILES.values.tolist()
    same_smiles = [smiles for smiles in smiles_list if smiles_list.count(smiles) > 1]
    new_df = df[~df.SMILES.isin(same_smiles)]
    df = df.drop(new_df.index)
    for smiles in same_smiles:
        if df.SMILES.isin([smiles]).values.std() <= 0.2:
            new_line = df[df.SMILES.isin([smiles])].iloc[0,]
            new_line.y = df.SMILES.isin([smiles]).values.mean()
            new_df.append(new_line)
    df_a = new_df
    return new_df

import time
class Logger():
    def __init__(self, columns, file, header=None, just='center' ,print=print):
        self.start_time = time.time()
        self.level = 0
        self.buffers = ""
        self.path = file
        self.align = {"center": '^', "left": '<', "right": '>'}[just]
        self.max_length = max(list(map(len, columns))+[6])+2
        columns = columns.copy() + ["runtime"]
        f = ""
        for _ in range(len(columns)):
            f += "{{:{}{}}} ".format(self.align,self.max_length)
        self.format = f.format
        if header is not None:
            print(header)
            self.buffers += header+'\n'
        print(self.format(*columns))
        self.buffers += self.format(*columns) + '\n'
        print(self.format(*["-"*(self.max_length-1)]*len(columns)))
        self.buffers += self.format(*["-"*(self.max_length-1)]*len(columns)) + '\n'

    def pf(self, *args):
        columns = [f"%.2f"%i if  isinstance(i, np.floating) or isinstance(i,float) else i for i in args]
        columns.append(self.getStrTime())
        print(self.format(*columns))
        self.buffers += self.format(*columns) + '\n'
        self.writer()

    def getStrTime(self):
        t1 = time.time()
        t = int(t1 - self.start_time)
        hour, min = 60*60, 60
        h ,m, s = t//hour, t%hour//min, t%hour%min
        return "{:0>2.0f}:{:0>2.0f}:{:0>2.0f}".format(h, m ,s)

    def writer(self):
        with open(self.path,'a') as f:
            f.write(self.buffers)
            self.buffers = ""
    def warning():
        pass
    def error():
        pass
    def info():
        pass

def convert_tabular(df_path,result_path,separator='|'):
    df = pd.read_csv(df_path, dtype=str)
    row_num = len(df.index)
    columns_num = len(df.columns)
    with open(result_path,'a')as f:
        f.write("\\begin{table}\n\\centering\n\\resizebox{\\textwidth}{30mm}{\n")
        f.write('\\begin{{tabular}}{{{}}}\n'.format(separator+'c|'+('c'+separator)*(columns_num-1)))
        f.write('\\hline\n')
        f.write('\\rowcolor[rgb]{0.8,.8,.8}&\multicolumn{' +str(columns_num-1)+'}{c}{FedAvg average epochs}\\\\\n')
        f.write(r'\rowcolor[rgb]{.8,.8,.8}\multirow{-2}{*}{Dataset(size)}')
        buffer = ''
        for i in df.columns:
            buffer += '&'+str(i)
        f.write(buffer.strip('&')+'\\\\\n')
        for i in range(row_num):
            f.write('\\hline\n')
            buffer = ''
            for j in range(columns_num):
                buffer += '&'+pre(df.iloc[i,j])
            f.write(buffer.strip('&')+'\\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}}\n')
        f.write(f"\caption{{{df_path.split('/')[-1].strip('.csv').replace('_',' ')}}}\n")
        f.write(r"\end{table}"+'\n\n')

def pre(s):
    import math
    if isinstance(s,float):
        if math.isnan(s) :
            return ' '
        else:
            return "%0.4f"%s    #{:0.4f}.format(s) or f"{'%0.4f'%s}"
    else:
        return s

def ecfp(smileslist=None,fp_radius=2,nbits=2048,embedding=None):
    ecfp_list =[]
    for index,smile  in enumerate(smileslist):
        mol = Chem.MolFromSmiles(smile)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius,nBits=nbits).ToBitString()
        ecfp_list.append([int(i) for i in fp])
    if embedding is not None:
        n, _ =np.array(ecfp_list).shape
        embedding = embedding.repeat(n, axis=0)
        ecfp_list = np.concatenate((np.array(ecfp_list),embedding), axis=1)
    return np.array(ecfp_list)

def check_smiles(smiles_list):
    """
    Processing NaN values
    Checking molecular legitimacy
    :param smiles_list: smiles list of molecules

    :return:
    """
    print("number of all smiles: ", len(smiles_list))

    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    del_smiles_list = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()
            #
            # atom_num_dist.append(len(mol.GetAtoms()))
            # Chem.SanitizeMol(mol)
            # Chem.DetectBondStereochemistry(mol, -1)
            # Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
            # Chem.AssignAtomChiralTagsFromStructure(mol, -1)
            # canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            remained_smiles.append(smiles)

        except:
            print('can not convert this {} smiles'.format(smiles))
            del_smiles_list.append(smiles)
    print("number of successfully processed smiles: ", len(remained_smiles))
    return del_smiles_list

def check_smiles(smiles_list):
    """
    Processing NaN values
    Checking molecular legitimacy
    :param smiles_list: smiles list of molecules

    :return:
    """
    print("number of all smiles: ", len(smiles_list))

    atom_num_dist = []
    remained_smiles = []
    canonical_smiles_list = []
    del_smiles_list = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString()
            #
            # atom_num_dist.append(len(mol.GetAtoms()))
            # Chem.SanitizeMol(mol)
            # Chem.DetectBondStereochemistry(mol, -1)
            # Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
            # Chem.AssignAtomChiralTagsFromStructure(mol, -1)
            # canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            remained_smiles.append(smiles)

        except:
            print('can not convert this {} smiles'.format(smiles))
            del_smiles_list.append(smiles)
    print("number of successfully processed smiles: ", len(remained_smiles))
    return del_smiles_list


def get_aqsol_data(path = "data/aqsoldb.csv", clientList = ['a','b','c','d'], seed = 1):
    """
    获取水溶性数据集

    Args:
        path (str, optional): 水溶性数据的路径. Defaults to "data/aqsoldb.csv".
        seed (int, optional): 随机种子. Defaults to 1.

    Returns:
        test_df (DataFrame): 测试集
        train_df (DataFrame): 训练集
    """
    df = pd.read_csv(path)
    train_data, test_data = {}, {}
    for name, group in df.groupby("user"):
        if name in clientList:
            train_data[name] = group.sample(frac=0.9, random_state=seed)
            group = group.drop(train_data[name].index)
        test_data[name] = group
    return train_data, test_data


def get_hERG_data(path = "data/hERG.csv", clientList = ["ChEMBL26", "PubChem-NCATS", "PubChem-JHICC", "Cai"], seed = 1):
    """
    获取hERG数据集

    Args:
        path (str, optional): hERG数据集路径. Defaults to "data/hERG.csv".
        seed (int, optional): 随机种子. Defaults to 1.

    Returns:
        test_df (DataFrame): 测试集
        train_df (DataFrame): 训练集

    """
    df = pd.read_csv(path)
    train_data, test_data = {}, {}
    for name, group in df.groupby("user"):
        if name in clientList:
            train_data[name] = group.sample(frac=0.9, random_state=seed)
            group = group.drop(train_data[name].index)
        test_data[name] = group
    return train_data, test_data


def get_kinase_data(path = "../data/kinase.csv", seed = 1, kinase = "AKT1"):
    """
    获取kinase数据集

    Args:
        path (str, optional): kinase数据集路径. Defaults to "data/hERG.csv".
        seed (int, optional): 随机种子. Defaults to 1.

    Returns:
        test_df (DataFrame): 测试集
        train_df (DataFrame): 训练集

    """
    df = pd.read_csv(path)
    df = df.rename({kinase: "y"}, axis=1)
    df = df.dropna(axis=0, subset=["y"])
    test_df = df.groupby("user").sample(frac=0.1, random_state=seed)
    df = df.drop(index=test_df.index)
    train_df = df.sample(frac=1, random_state=seed)
    return test_df, train_df


def get_client_list(df):
    """
    获取参与训练的客户列表

    Args:
        df (DataFrame): 输入初始读入的数据集

    Returns:
        client_list (List): 客户端列表 
    """
    return list(df["user"].unique())


def df2xy(df):
    """
    将DataFrame转化为网络输入的特征和标签

    Args:
        df (DataFrame): 数据集

    Returns:
        x (array): 特征
        y (array): 标签
    """
    x = df[["x"+str(i) for i in range(2048)]].values
    y = df["y"].values
    return torch.Tensor(x), torch.Tensor(y)

def get_k_fold_data(df, k = 5, seed = 1):
    """
    将数据集划分为k份

    Args:
        df (DataFrame): 数据集
        k (int, optional): 划分的折数. Defaults to 5.
        seed (int, optional): 随机种子. Defaults to 1.

    Returns:
        k_fold_df_list (List[DataFrame, ...]): 包含k个DataFrame的列表
    """
    n = len(df) // 5
    k_fold_df_list = []
    for i in range(k):
        k_fold_df_list.append(df.iloc[i*n:(i+1)*n, :])
    return k_fold_df_list

class MseLossNan(nn.Module):
    def __init__(self):
        super(MseLossNan, self).__init__()

    def forward(self, input, target, reduction="sum"):
        if not (target.size() == target.size()):
            print("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(target.size(), target.size()),)
        loss = (target - input)
        loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)

