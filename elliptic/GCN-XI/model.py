import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from config import args
import math
import numpy as np
import torch
from dataset import EllipticDataset

def statistical_information_XA(dataset:EllipticDataset):
    """
    return the mean and varience of the whole dataset
    :param dataset: type is Xdataset,contain a number of torch.tensor
    :return:total_mean,total_var
    """
    n,mu,sigma2=[],[],[]
    for i in range(dataset.__len__()):
        A,_,_=dataset.__getitem__(i)
        n.append(A.shape[0])
        mu.append(A.mean().item())
        sigma2.append(A.var().item())
    np_n=np.array(n).astype(float)
    np_mu=np.array(mu)
    np_sigma2=np.array(sigma2)
    sum_=sum(n)
    total_mean=np.dot(np_n,np_mu)/sum_
    total_var=(np.dot(np_n,np_sigma2)+np.dot(np_n,np_mu**2)-sum(sigma2)-sum_*total_mean**2)/(sum_-1)
    return total_mean,total_var

def statistical_information_XF_bycol(dataset:EllipticDataset):
    n,mu,sigma2=[],[],[]
    total_mean,total_var=[],[]
    for i in range(dataset.__len__()):
        _,F,_=dataset.__getitem__(i)
        if i==0:
            col_num=F.shape[1]
        for j in range(col_num):
            if len(n)<col_num:
                n.append([])
            if len(mu)<col_num:
                mu.append([])
            if len(sigma2)<col_num:
                sigma2.append([])
            n[j].append(F.shape[0])
            mu[j].append(F[:,j].mean().item())
            sigma2[j].append(F[:,j].var().item())
    for i in range(col_num):
        np_n=np.array(n[i]).astype(float)
        np_mu=np.array(mu[i])
        np_sigma2=np.array(sigma2[i])
        sum_=sum(n[i])
        total_mean.append(np.dot(np_n,np_mu)/sum_)
        total_var.append((np.dot(np_n,np_sigma2)+np.dot(np_n,np_mu**2)-sum(np_sigma2)-sum_*total_mean[-1]**2)/(sum_-1))
    return total_mean,total_var
def statistical_information_X(dataset:EllipticDataset):
    nA,muA,sigma2A=[],[],[]
    nF,muF,sigma2F=[],[],[]
    total_meanF,total_varF=[],[]
    for i in range(dataset.__len__()):
        A,F,_=dataset.__getitem__(i)
        F = torch.nn.functional.normalize(F, p=1, dim=0)
        nA.append(A.shape[0])
        muA.append(A.mean().item())
        sigma2A.append(A.var().item())
        if i==0:
            col_num=F.shape[1]
        for j in range(col_num):
            if len(nF)<col_num:
                nF.append([])
            if len(muF)<col_num:
                muF.append([])
            if len(sigma2F)<col_num:
                sigma2F.append([])
            nF[j].append(F.shape[0])
            muF[j].append(F[:,j].mean().item())
            sigma2F[j].append(F[:,j].var().item())
    for i in range(col_num):
        np_nF=np.array(nF[i]).astype(float)
        np_muF=np.array(muF[i])
        np_sigma2F=np.array(sigma2F[i])
        sum_F=sum(nF[i])
        total_meanF.append(np.dot(np_nF,np_muF)/sum_F)
        total_varF.append((np.dot(np_nF,np_sigma2F)+np.dot(np_nF,np_muF**2)-sum(np_sigma2F)-sum_F*total_meanF[-1]**2)/(sum_F-1))
    np_nA = np.array(nA).astype(float)
    np_muA = np.array(muA)
    np_sigma2A = np.array(sigma2A)
    sum_A = sum(nA)
    total_mean = np.dot(np_nA, np_muA) / sum_A
    total_var = (np.dot(np_nA, np_sigma2A) + np.dot(np_nA, np_muA ** 2) - sum(
        sigma2A) - sum_A * total_mean ** 2) / (sum_A - 1)
    return total_mean,total_var,total_meanF,total_varF

def xinitialization(dataset:EllipticDataset, n:int,d:int):
    """

    :param dataset: type Xdataset
    :param n: the number of nodes
    :param d: the dimension of hidden feature
    :return: torch.tensor that shape is determined by parameter created by Xinitialization method
    """
    total_mean,total_var=statistical_information_X(dataset)
    x=torch.normal(mean=0,std=math.sqrt(2./(d*n*total_var+d*total_mean**2)),size=(d,d))
    return x
def xinitialization(mean:float,var:float,n:int,d:int):
    x = torch.normal(mean=0, std=math.sqrt(2. / (d * n * var + d * mean ** 2)),size=(d,d))
    return x





class GCN(nn.Module):
    def __init__(self,
                 mean: float, var: float, n: int, d: int,
                 original_feature_dimension,
                 root_enhance_parameter=0.,
                 residual_propagation_parameter=0.,
                 activation='ReLu',
                 initialization_method='Xinitialization'):
        """
        :param mean: the mean of test dataset (the estimate mean of the whole distribution)
        :param var: the variance of test dataset (the estimate variance of the whole distribution)
        :param n: the number of nodes
        :param d: the dimension of hidden feature
        :param initialization_method: 'Xinitialization','xavier_normal' or 'kaiming_normal'
        """
        super(GCN, self).__init__()
        self.n = n
        self.d = d
        self.original_feature_dimension = original_feature_dimension
        self.alpha = root_enhance_parameter
        self.beta = residual_propagation_parameter
        self.root_enhance_layer = self.liner
        self.L = nn.Linear(self.original_feature_dimension,
                           self.d)
        #self.activation = self.XReLU

        if activation == 'LeakyReLu':
            self.activation = nn.LeakyReLU()
        elif activation == 'PReLu':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()

        if initialization_method == 'Xinitialization':
            self.W = nn.Parameter(
                xinitialization(mean=mean, var=var, n=self.n, d=self.d))
        elif initialization_method == 'xavier_normal':
            self.W = nn.init.xavier_normal_(torch.Tensor(d, d))
        else:
            self.W = nn.init.kaiming_normal_(torch.Tensor(d, d))

    def forward(self, A: torch.tensor, F: torch.tensor,
                original_F: torch.tensor, layer_sn: int):
        """
        :param A: adjacency matrix
        :param F: feature matrix (output of previous layer or original feature matrix)
        :param original_F: original feature matrix
        :param layer_sn: the serial number of the layer
        :return: output tensor
        """
        res = F
        if layer_sn == 0:
            x = self.liner(F)
        else:
            x = F
        if self.alpha != 0:
            if layer_sn == 0:
                x += self.activation(self.alpha * self.liner(
                    torch.cat(original_F[:, 0, :],
                              torch.zeros(original_F.shape[1] - 1,
                                          original_F.shape[2]),
                              dim=1)))
            else:
                x += self.activation(self.alpha * torch.cat(original_F[:, 0, :],
                                                            torch.zeros(
                                                                original_F.shape[
                                                                    1] - 1,
                                                                original_F.shape[
                                                                    2])))
        if layer_sn == 0:
            return torch.matmul(torch.matmul(A, x), self.W)
        if self.beta == 0. or res.shape != x.shape:
            return torch.matmul(torch.matmul(A, x), self.W)
        else:
            x = torch.matmul(torch.matmul(A, x), self.W)
            x += self.activation(self.beta * res)
            return x

    def liner(self, x):
        """
        x为3维张量
        :param x: x.shape=(*,*,self.original_feature_dimension)
        :return:
        """
        c = []
        for i in range(x.shape[0]):
            c.append(self.L(x[i]))
        s = torch.stack(c, dim=0)
        return s

    def XReLU(self, x):
        return torch.max(x, -torch.full_like(x, 3 * math.sqrt(self.var)))

class XIGCNL(nn.Module):
    def __init__(self,
                 mean: float, var: float, n: int, d: int,
                 original_feature_dimension,
                 layers_number: int,
                 activation='ReLu',
                 initialization_method='Xinitialization',
                 root_enhance_parameter=0.,
                 residual_propagation_parameter=0.):
        """

        :param mean:
        :param var:
        :param n:
        :param d:
        :param original_feature_dimension:
        :param layers_number:
        :param activation: 'LeakyRelu' ,'PRelu' or 'ReLu
        :param initialization_method:
        :param root_enhance_parameter:
        :param residual_propagation_parameter:
        """
        super(XIGCNL, self).__init__()
        self.mean = mean
        self.var = var
        self.n = n
        self.d = d
        self.original_feature_dimension = original_feature_dimension
        self.layers_number = layers_number
        self.alpha = root_enhance_parameter
        self.beta = residual_propagation_parameter
        self.layers = []
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)  # 从第1维开始打平，输出的是2维张量
        self.liner = nn.Linear(args.graph_size * self.d, 2)
        #self.activation = self.XReLU

        if activation == 'LeakyReLu':
            self.activation = nn.LeakyReLU()
        elif activation == 'PReLu':
            self.activation = nn.PReLU()
        else:
            self.activation = nn.ReLU()

        for i in range(self.layers_number):
            self.layers.append(GCN(self.mean, self.var, self.n, self.d,
                                   self.original_feature_dimension,
                                   self.alpha,
                                   self.beta,
                                   activation,
                                   initialization_method))
        print('GCN-kaiming information:','self.mean:',self.mean,
              '\nself.var:',self.var,'\nself.n:',self.n,
              '\nself.d:',self.d,
              '\nself.original_feature_dimension:',self.original_feature_dimension,
              '\nself.layers_number:',self.layers_number,
              '\nself.alpha:',self.alpha,
              '\nself.beta:',self.beta,
              '\nactivation:',activation)

    def forward(self, A: torch.Tensor, F: torch.Tensor):
        x = self.activation(self.layers[0](A, F, F, 0))
        for i in range(1, self.layers_number):
            x = self.activation(self.layers[i](A, x, original_F=F, layer_sn=i))
        #x = x + self.mean
        x = self.flatten(x)
        x = self.activation(self.fc(x))
        return x

    def fc(self, x):
        """
        x为3维张量
        :param x: x.shape=(*,*,self.original_feature_dimension)
        :return:
        """
        c = []
        for i in range(x.shape[0]):
            c.append(self.liner(x[i]))
        s = torch.stack(c, dim=0)
        return s

    def XReLU(self, x):
        return torch.max(x, -torch.full_like(x, 3 * math.sqrt(self.var)))