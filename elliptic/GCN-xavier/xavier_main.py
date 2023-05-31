"""Torch modules for graph convolutions(GCN-kaiming)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import argparse
import sys
import time

from experiment import Experiment
from model_dgl import GCN
from dataset_dgl import xdataset_,train_dataset
import torch
from config import Logger



recordname = '../log/GCN-xavier'+time.strftime('%Y_%m_%d_%H%M%S',time.localtime())+'.txt'
sys.stdout = Logger(recordname, sys.stdout)

args = argparse.ArgumentParser()
args.add_argument('--learning_rate', type=float, default=3e-4)
args.add_argument('--epochs', type=int, default=150)
args.add_argument('--batch_size', type=int, default=24)
args.add_argument('--train_set_len', type=float, default=0.8)
args.add_argument('--shuffle', type=bool, default=True)
args.add_argument('--graph_size', type=int, default=50)
args = args.parse_args(args=[])
print(args)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('model GCN-xavier:')
print('layer_num:',2)
model = GCN(in_dim=166,
            out_dim=2,
            graph_size=args.graph_size,
            layer_num=2
            ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

criterion = torch.nn.CrossEntropyLoss()

exp = Experiment(model, optimizer, criterion, xdataset_, args)
aloss = float('Inf')
mark = 0
for epoch in range(args.epochs):
    exp.train(epoch)
    exp.test()
