import argparse
import math

from dataset_dgl import xdataset_
from experiment import Experiment
from model_dgl import statistical_information_X
import torch
import torch.nn.functional as F
import torch.nn as nn

def xinitialization(self, mean: float, var_A: float, var_F: list, n: int, d: int):
    std_list = []
    for i in var_F:
        std_list.append(
            math.sqrt(2. * i / (d * n * var_A * i + d * mean ** 2)))
    std = torch.tensor(std_list).mean().item()
    x = nn.Parameter(torch.normal(mean=0, std=std, size=(d, d)))
    return x

class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features,mean_A,var_A,var_F,num_of_nodes):
        super(GCNLayer, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.mean=mean_A
        self.var=var_A
        self.var_F=var_F
        self.num_of_nodes=num_of_nodes
        self._out_feats=out_features
        # 初始化权重和偏差
        self.weight=xinitialization(self.mean, self.var,self.var_F,
                                          self.num_of_nodes, self._out_feats,in_features)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x, adj_matrix):
        # 计算度矩阵和规范化邻接矩阵
        degree = torch.sum(adj_matrix, axis=1)
        degree_matrix = torch.diag(degree)
        normalized_adj_matrix = torch.matmul(torch.matmul(torch.inverse(degree_matrix), adj_matrix), torch.inverse(degree_matrix))

        # 计算GCN层输出
        x = torch.matmul(normalized_adj_matrix, x)
        x = torch.matmul(x, self.weight) + self.bias
        x = F.relu(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,mean_A,
            var_A,
            var_F,num_of_nodes):
        super(GCN, self).__init__()
        self.input_layer = GCNLayer(input_dim, hidden_dim,mean_A,var_A,var_F,num_of_nodes)
        self.hidden_layer = GCNLayer(hidden_dim, output_dim,mean_A,var_A,var_F,num_of_nodes)

    def forward(self, x, adj_matrix):
        x = self.input_layer(x, adj_matrix)
        print(torch.var(self.input_layer.weight))
        x = self.hidden_layer(x, adj_matrix)
        print(torch.var(self.hidden_layer.weight))
        return x
args = argparse.ArgumentParser()
args.add_argument('--learning_rate', type=float, default=3e-4)
args.add_argument('--epochs', type=int, default=150)
args.add_argument('--batch_size', type=int, default=24)
args.add_argument('--train_set_len', type=float, default=0.8)
args.add_argument('--shuffle', type=bool, default=True)
args.add_argument('--graph_size', type=int, default=50)
args.add_argument('--LRM_layer_num',type=int,default=1)
args.add_argument('--LRM_dimension',type=int,default=498)
args.add_argument('--layer_num',type=int,default=3)
args.add_argument('--alpha',type=float,default=0.2)
args.add_argument('--beta',type=float,default=0.5)
args.add_argument('--gamma',type=float,default=0.5)
args.add_argument('--delta',type=float,default=0.8)
args = args.parse_args(args=[])
print(args)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_len=int(0.8*xdataset_.__len__())
test_len=xdataset_.__len__()-train_len
train_set,test_set=torch.utils.data.random_split(xdataset_,[train_len,test_len])
total_mean, total_var, total_mean_F, total_var_F = statistical_information_X(
    train_set)
model = GCN(166,
            166,
            2,
            mean_A=total_mean,
            var_A=total_var,
            var_F=total_var_F,
            num_of_nodes=50).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

criterion = torch.nn.CrossEntropyLoss()

exp = Experiment(model, optimizer, criterion,train_set,test_set,args)

for epoch in range(args.epochs):
    exp.train(epoch)
    exp.test()

