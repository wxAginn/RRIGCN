import argparse
import sys
import torch
from config import  sk_preformance, Logger, newperformance
from dataload.Xdataset import Xdataset
from torch.utils.data import random_split
from model_dgl import GAT
from experiment import Experiment

recordname = '../log/GAT_2021_11_25_2.txt'
sys.stdout = Logger(recordname, sys.stdout)
args = argparse.ArgumentParser()
args.add_argument('--learning_rate', type=float, default=3e-4)
args.add_argument('--epochs', type=int, default=1000)
args.add_argument('--batch_size', type=int, default=24)
args.add_argument('--train_set_len', type=float, default=0.8)
args.add_argument('--shuffle', type=bool, default=True)
args.add_argument('--graph_size', type=int, default=3000)
args = args.parse_args(args=[])
print(args)
xdataset_ = Xdataset('../dataset')
t_len = int(xdataset_.__len__() * 0.8)
train_dataset, test_dataset = random_split(xdataset_, [t_len, xdataset_.__len__() - t_len])

model = GAT(6, 2, num_heads=3)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
optimizer_comp = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

criterion = torch.nn.CrossEntropyLoss()

exp = Experiment(model, optimizer, criterion, train_dataset,test_dataset, args)

for epoch in range(args.epochs):
    exp.train(epoch)
    exp.test()