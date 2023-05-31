"""Torch modules for graph convolutions(GCN)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import argparse
import sys
from experiment import Experiment
from model_dgl import GCN
from dataload.Xdataset import Xdataset
import torch
from GAT.config import Logger
from torch.utils.data import random_split

xdataset_ = Xdataset('../dataset')
recordname = '../log/GCN_xavier_2021_10_4_LeakyReLU5.txt'
sys.stdout = Logger(recordname, sys.stdout)
t_len = int(xdataset_.__len__() * 0.8)
train_dataset, test_dataset = random_split(xdataset_, [t_len, xdataset_.__len__() - t_len])

'''
total_mean, total_var=0.00036874049272486774,0.0003684817044070348
total_mean_F=[1.1443002322467163e-06, 0.00019547124770309362, 0.05324531916413449, 1.0727070165340593e-06, 1.0633775664653203e-06, 1.6239820931536743e-09]
total_var_F=[2.978577954565642e-11, 7.011216620605938e-07, 0.05021227163285519, 2.8373926185118726e-11, 2.7991845859231312e-11, 1.2018615241194142e-13]
#print(total_mean, '\n', total_var, '\n', total_mean_F, '\n', total_var_F)
'''
args = argparse.ArgumentParser()
args.add_argument('--learning_rate', type=float, default=3e-4)
args.add_argument('--epochs', type=int, default=150)
args.add_argument('--batch_size', type=int, default=24)
args.add_argument('--train_set_len', type=float, default=0.8)
args.add_argument('--shuffle', type=bool, default=True)
args.add_argument('--graph_size', type=int, default=3000)
args = args.parse_args(args=[])
print(args)

print('model GCN-XI:')
print('layer_num:', 2)
model = GCN(in_dim=6,
            out_dim=2,
            graph_size=args.graph_size,
            layer_num=2)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

criterion = torch.nn.CrossEntropyLoss()

exp = Experiment(model, optimizer, criterion, train_dataset,test_dataset, args)

for epoch in range(args.epochs):
    exp.train(epoch)
    exp.test()
