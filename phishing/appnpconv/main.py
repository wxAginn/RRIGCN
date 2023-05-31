"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import argparse
import sys
import time
sys.path.append('./')
sys.path.append('../')
from experiment import Experiment
from dgl.nn.pytorch.conv import GATConv
from dataload.Xdataset import Xdataset
from torch.utils.data import Dataset
import torch
from dgl.nn.pytorch.conv.appnpconv import APPNPConv
from GAT.config import Logger
from model_dgl import statistical_information_X
from torch.utils.data import random_split
from torch_geometric.data import Data
import torch.nn as nn
import random
random.seed(700)
torch.manual_seed(700)
class APPNP(nn.Module):
    def __init__(self,in_feat,out_feat,k,alpha):
        super(APPNP, self).__init__()
        self.conv1=GATConv(in_feat,out_feat,3)
        self.conv2=APPNPConv(k=k,alpha=alpha)
        self.fc = nn.Sequential(nn.Flatten(0, -1),
                                nn.Linear(out_feat*500*3, 2))

    def forward(self,A,x):
        x=self.normalize(x)
        out=self.conv1(A,x)
        out=self.conv2(A,out)
        out=self.fc(out)
        return out

    def normalize(self, X: torch.TensorType):
        N = torch.nn.functional.normalize(X, p=1, dim=0)
        '''
        std = []
        for i in range(N.shape[1]):
            if math.sqrt(N[:, i].var()) <= 0. or self.var_F[i] <= 0.:
                std.append(1.)
            else:
                # std.append( 1. / math.sqrt(self.var_F[i]))
                std.append(1. / math.sqrt(N[:, i].var()))
        std = torch.diag(torch.tensor(std))
        M = torch.matmul(N, std)
        '''
        return N


xdataset_ = Xdataset('../dataset/')
#recordname = '../log/GCN-XI'+time.strftime('%Y_%m_%d_%H%M%S',time.localtime())+'.txt'
#sys.stdout = Logger(recordname, sys.stdout)
t_len = int(xdataset_.__len__() * 0.8)
print(xdataset_.__len__())
train_dataset, test_dataset = random_split(xdataset_, [t_len, xdataset_.__len__() - t_len])
print(train_dataset.__len__())
print(test_dataset.__len__())
'''
class dgl_to_pygdataset(Dataset):
    def __init__(self,dataset:Dataset):
        self.dgl_dataset=dataset
    def __getitem__(self, item):
        g,F,L=self.dgl_dataset.__getitem__(item)
        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        node_features = g.ndata['feature']  # 假设节点特征存储在'feature'中
        edge_features = g.edata['feature']  # 假设边特征存储在'feature'中
        pyg_graph = Data(
            x=node_features,
            edge_index=torch.tensor(g.edges()).t().contiguous(),
            edge_attr=edge_features,
            num_nodes=num_nodes,
            num_edges=num_edges
        )
        

train_dataset,test_dataset=dgl_to_pygdataset(train_dataset),dgl_to_pygdataset(test_dataset)
print(train_dataset.__getitem__(0))
'''
'''
total_mean, total_var=0.00036874049272486774,0.0003684817044070348
total_mean_F=[1.1443002322467163e-06, 0.00019547124770309362, 0.05324531916413449, 1.0727070165340593e-06, 1.0633775664653203e-06, 1.6239820931536743e-09]
total_var_F=[2.978577954565642e-11, 7.011216620605938e-07, 0.05021227163285519, 2.8373926185118726e-11, 2.7991845859231312e-11, 1.2018615241194142e-13]
#print(total_mean, '\n', total_var, '\n', total_mean_F, '\n', total_var_F)
'''
args = argparse.ArgumentParser()
args.add_argument('--learning_rate', type=float, default=3e-4)
args.add_argument('--epochs', type=int, default=10)
args.add_argument('--batch_size', type=int, default=2)
args.add_argument('--train_set_len', type=float, default=0.8)
args.add_argument('--shuffle', type=bool, default=True)
args.add_argument('--graph_size', type=int, default=500)
args = args.parse_args(args=[])
print(args)

model = APPNP(in_feat=6,out_feat=2,k=3,alpha=0.8)
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

criterion = torch.nn.CrossEntropyLoss()

exp = Experiment(model, optimizer, criterion, train_dataset,test_dataset, args)

for epoch in range(args.epochs):
    exp.train(epoch)
    exp.test()
